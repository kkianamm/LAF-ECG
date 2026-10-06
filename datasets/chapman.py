from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

# Reuse the exact generic classification/normalization/resampling implementation
# that your PTB-XL pipeline already uses.
from .ptbxl import ClassificationDataset


# Standard 12-lead order expected by MedTsLLM/MOMENT.
LEAD_ORDER = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

# Rhythm SNOMED-CT codes listed by the PhysioNet release.
# SAAWR and WAVN share 195101003 in ConditionNames_SNOMED-CT.csv;
# we use the dataset acronym SAAWR as the canonical name.
SNOMED_TO_RHYTHM: Dict[int, str] = {
    426177001: "SB",
    426783006: "SR",
    164889003: "AFIB",
    427084000: "ST",
    164890007: "AF",
    427393009: "SA",
    426761007: "SVT",
    713422000: "AT",
    233896004: "AVNRT",
    233897008: "AVRT",
    195101003: "SAAWR",
}

# Default used for your Chapman experiment.
CHAPMAN_8_CLASSES = [
    "SB",
    "SR",
    "AFIB",
    "ST",
    "AF",
    "SA",
    "SVT",
    "AT",
]

CHAPMAN_9_CLASSES = [
    "SB", "SR", "AFIB", "ST", "AF", "SA", "SVT", "AT", "AVRT"
]

# Optional broader rhythm-only setup supported by the same loader.
CHAPMAN_11_CLASSES = [
    "SB", "SR", "AFIB", "ST", "AF", "SA",
    "SVT", "AT", "AVNRT", "AVRT", "SAAWR"
]


def _cfg_get(obj, key, default=None):
    """Works with dicts and the repository's dict-to-object config wrapper."""
    if obj is None:
        return default
    if hasattr(obj, "get"):
        try:
            return obj.get(key, default)
        except TypeError:
            pass
    return getattr(obj, key, default)


def _parse_comment_fields(comments: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for comment in comments or []:
        if ":" not in comment:
            continue
        key, value = comment.split(":", 1)
        out[key.strip().lower()] = value.strip()
    return out


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if value.lower() in {"nan", "na", "n/a", "none", "unknown", ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _record_number(record_name: str) -> Optional[int]:
    """Extract JS01234 -> 1234. Returns None for non-JS names."""
    match = re.fullmatch(r"JS(\d+)", record_name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


class ChapmanClassificationDataset(ClassificationDataset):
    """
    Single-label rhythm classification for PhysioNet ecg-arrhythmia v1.0.0.

    The implementation intentionally mirrors the PTB-XL classification path:
      * [N, T, 12] ECG arrays
      * training-split StandardScaler normalization
      * deterministic integer class labels
      * x_enc produced by the existing resample_to_history() method
      * text descriptions contain metadata only (never the diagnosis label)

    Chapman-specific differences:
      * WFDB signals are 500 Hz, 10 s.
      * We anti-alias downsample 500 -> target_fs (default 100 Hz) BEFORE the
        existing history_len resampling, so PTB-XL and Chapman have the same
        1,000-sample full-record representation when target_fs=100.
      * The PhysioNet release has no PTB-XL-style official 10-fold partition,
        so we make and persist a deterministic stratified 70/10/20 manifest.
    """

    supported_tasks = ["classification"]
    supported_features = ["M"]
    description = (
        "A 12-lead resting ECG arrhythmia dataset from the PhysioNet "
        "ecg-arrhythmia release. Each recording is 10 seconds long; source "
        "signals are sampled at 500 Hz and are converted to 100 Hz by this "
        "experiment before MedTsLLM/MOMENT processing."
    )
    task_description = (
        "Classify the entire 12-lead ECG sequence into one of the configured "
        "cardiac rhythm classes."
    )
    sampling_frequency = 500

    def __init__(self, config, split: str):
        dataset_name = config.data.dataset
        datasets_cfg = _cfg_get(config, "datasets", {})
        cfg = _cfg_get(datasets_cfg, dataset_name, {})
        self.chapman_cfg = cfg

        label_set = str(_cfg_get(cfg, "label_set", "eight")).lower()
        if label_set in {"eight", "8", "8class", "8-class"}:
            self.label_set = "eight"
            self.class_names = list(CHAPMAN_8_CLASSES)
        elif label_set in {"nine", "9", "9class", "9-class"}:
            self.label_set = "nine"
            self.class_names = list(CHAPMAN_9_CLASSES)
        elif label_set in {"all", "eleven", "11", "11class", "11-class"}:
            self.label_set = "eleven"
            self.class_names = list(CHAPMAN_11_CLASSES)
        else:
            raise ValueError(
                f"Unknown Chapman label_set={label_set!r}. "
                "Use 'eight', 'nine', or 'eleven'."
            )

        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.n_classes = len(self.class_names)

        self.target_fs = int(_cfg_get(cfg, "target_fs", 100))
        self.split_seed = int(_cfg_get(cfg, "split_seed", 0))
        self.train_fraction = float(_cfg_get(cfg, "train_fraction", 0.70))
        self.val_fraction = float(_cfg_get(cfg, "val_fraction", 0.10))
        self.test_fraction = float(_cfg_get(cfg, "test_fraction", 0.20))

        total = self.train_fraction + self.val_fraction + self.test_fraction
        if not math.isclose(total, 1.0, abs_tol=1e-8):
            raise ValueError(
                "train_fraction + val_fraction + test_fraction must equal 1.0; "
                f"got {total:.6f}"
            )
        if self.target_fs <= 0:
            raise ValueError("target_fs must be positive.")

        # Optional reproducible cohort restriction. 0 means "no bound".
        self.record_id_min = int(_cfg_get(cfg, "record_id_min", 0) or 0)
        self.record_id_max = int(_cfg_get(cfg, "record_id_max", 0) or 0)

        super().__init__(config, split)

    # ------------------------------------------------------------------
    # Paths / manifest
    # ------------------------------------------------------------------
    def _resolve_root(self) -> Path:
        configured = str(_cfg_get(self.chapman_cfg, "root", "data/chapman_shaoxing"))
        root = Path(configured).expanduser()

        if not root.is_absolute():
            repo_root = Path(__file__).resolve().parents[1]
            root = (repo_root / root).resolve()

        # Support either:
        #   data/chapman_shaoxing/WFDBRecords
        # or
        #   data/chapman_shaoxing/ecg-arrhythmia-1.0.0/WFDBRecords
        candidates = [
            root,
            root / "ecg-arrhythmia-1.0.0",
            root / "ecg-arrhythmia",
        ]
        for candidate in candidates:
            if (candidate / "WFDBRecords").is_dir():
                return candidate

        raise FileNotFoundError(
            "Could not find WFDBRecords. Expected it under one of:\n  - "
            + "\n  - ".join(str(c / "WFDBRecords") for c in candidates)
            + "\nDownload/extract PhysioNet ecg-arrhythmia v1.0.0 first."
        )

    def _manifest_path(self, root: Path) -> Path:
        split_token = (
            f"tr{self.train_fraction:.4f}_va{self.val_fraction:.4f}_"
            f"te{self.test_fraction:.4f}"
        ).replace(".", "p")
        cohort_token = f"min{self.record_id_min}_max{self.record_id_max}"
        return root / (
            f"medtsllm_manifest_{self.label_set}_seed{self.split_seed}_"
            f"{split_token}_{cohort_token}.csv"
        )

    def _cache_paths(self, root: Path, split: str) -> Dict[str, Path]:
        cache_dir = root / "medtsllm_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        split_token = (
            f"tr{self.train_fraction:.4f}_va{self.val_fraction:.4f}_"
            f"te{self.test_fraction:.4f}"
        ).replace(".", "p")
        cohort_token = f"min{self.record_id_min}_max{self.record_id_max}"
        prefix = (
            f"{self.label_set}_{self.target_fs}hz_seed{self.split_seed}_"
            f"{split_token}_{cohort_token}_{split}"
        )
        return {
            "data": cache_dir / f"{prefix}_data.npy",
            "labels": cache_dir / f"{prefix}_labels.npy",
            "descriptions": cache_dir / f"{prefix}_descriptions.json",
        }

    def _record_allowed(self, record_name: str) -> bool:
        rid = _record_number(record_name)
        if rid is None:
            return True
        if self.record_id_min > 0 and rid < self.record_id_min:
            return False
        if self.record_id_max > 0 and rid > self.record_id_max:
            return False
        return True

    def _build_manifest(self, root: Path, manifest_path: Path) -> pd.DataFrame:
        wfdb_root = root / "WFDBRecords"
        headers = sorted(wfdb_root.rglob("*.hea"))
        if not headers:
            raise FileNotFoundError(f"No .hea files found below {wfdb_root}")

        rows: List[dict] = []

        print(
            f"[Chapman] scanning {len(headers):,} WFDB headers "
            f"(label_set={self.label_set})..."
        )
        for header_path in tqdm(headers, desc="Scanning Chapman headers"):
            record_name = header_path.stem
            if not self._record_allowed(record_name):
                continue

            record_stem = header_path.with_suffix("")
            try:
                header = wfdb.rdheader(str(record_stem))
            except Exception as exc:
                print(f"[Chapman] warning: cannot read {record_stem}: {exc}")
                continue

            fields = _parse_comment_fields(getattr(header, "comments", []))
            dx_raw = fields.get("dx", "")
            try:
                dx_codes = [
                    int(x.strip()) for x in dx_raw.split(",") if x.strip()
                ]
            except ValueError:
                continue

            # Enforce clean single-rhythm supervision across all known rhythm codes.
            all_rhythms = []
            for code in dx_codes:
                rhythm = SNOMED_TO_RHYTHM.get(code)
                if rhythm is not None and rhythm not in all_rhythms:
                    all_rhythms.append(rhythm)

            if len(all_rhythms) != 1:
                continue

            rhythm = all_rhythms[0]
            if rhythm not in self.class_to_idx:
                continue

            fs = float(getattr(header, "fs", self.sampling_frequency))
            sig_len = int(getattr(header, "sig_len", 0) or 0)
            n_sig = int(getattr(header, "n_sig", 0) or 0)

            if n_sig != 12:
                continue

            age = _safe_float(fields.get("age"))
            sex = fields.get("sex", "Unknown")

            rows.append(
                {
                    "record": record_stem.relative_to(root).as_posix(),
                    "record_name": record_name,
                    "rhythm": rhythm,
                    "label": self.class_to_idx[rhythm],
                    "age": age,
                    "sex": sex,
                    "fs": fs,
                    "sig_len": sig_len,
                    "n_sig": n_sig,
                }
            )

        if not rows:
            raise RuntimeError(
                "No eligible single-rhythm Chapman records were found. "
                "Check the dataset root and label_set."
            )

        df = pd.DataFrame(rows).reset_index(drop=True)

        # Deterministic stratified 70/10/20 (configurable), persisted permanently.
        indices = np.arange(len(df))
        labels = df["label"].to_numpy()

        outer = StratifiedShuffleSplit(
            n_splits=1,
            train_size=self.train_fraction,
            random_state=self.split_seed,
        )
        train_idx, temp_idx = next(outer.split(indices, labels))

        temp_labels = labels[temp_idx]
        val_share_of_temp = self.val_fraction / (
            self.val_fraction + self.test_fraction
        )

        inner = StratifiedShuffleSplit(
            n_splits=1,
            train_size=val_share_of_temp,
            random_state=self.split_seed + 1,
        )
        val_rel_idx, test_rel_idx = next(
            inner.split(np.arange(len(temp_idx)), temp_labels)
        )
        val_idx = temp_idx[val_rel_idx]
        test_idx = temp_idx[test_rel_idx]

        df["split"] = ""
        df.loc[train_idx, "split"] = "train"
        df.loc[val_idx, "split"] = "val"
        df.loc[test_idx, "split"] = "test"

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(manifest_path, index=False)

        print(f"[Chapman] wrote split manifest: {manifest_path}")
        print("[Chapman] class counts by split:")
        counts = (
            df.groupby(["rhythm", "split"])
            .size()
            .unstack(fill_value=0)
            .reindex(self.class_names)
        )
        print(counts.to_string())
        return df

    def _load_manifest(self, root: Path) -> pd.DataFrame:
        path = self._manifest_path(root)
        if path.exists():
            df = pd.read_csv(path)
            # Guard against accidental stale/edited manifests.
            expected = {
                "record", "record_name", "rhythm", "label",
                "age", "sex", "fs", "sig_len", "n_sig", "split"
            }
            missing = expected.difference(df.columns)
            if missing:
                raise RuntimeError(
                    f"Manifest {path} is missing columns: {sorted(missing)}"
                )
            return df
        return self._build_manifest(root, path)

    # ------------------------------------------------------------------
    # ECG loading / anti-alias downsampling
    # ------------------------------------------------------------------
    def _fix_length(self, signal: np.ndarray, expected_len: int) -> np.ndarray:
        if signal.shape[0] == expected_len:
            return signal
        old_x = np.linspace(0.0, 1.0, signal.shape[0], endpoint=True)
        new_x = np.linspace(0.0, 1.0, expected_len, endpoint=True)
        out = np.empty((expected_len, signal.shape[1]), dtype=np.float32)
        for lead in range(signal.shape[1]):
            out[:, lead] = np.interp(new_x, old_x, signal[:, lead])
        return out

    def _load_record(self, root: Path, rel_stem: str) -> np.ndarray:
        stem = root / rel_stem
        signal, fields = wfdb.rdsamp(str(stem))
        signal = np.asarray(signal, dtype=np.float32)

        sig_names = list(fields.get("sig_name", []))
        if len(sig_names) != 12:
            raise RuntimeError(f"{stem}: expected 12 leads, got {len(sig_names)}")

        # Canonicalize lead ordering explicitly.
        name_to_idx = {name: i for i, name in enumerate(sig_names)}
        missing = [lead for lead in LEAD_ORDER if lead not in name_to_idx]
        if missing:
            raise RuntimeError(f"{stem}: missing leads {missing}")
        signal = signal[:, [name_to_idx[lead] for lead in LEAD_ORDER]]

        source_fs = int(round(float(fields.get("fs", self.sampling_frequency))))
        if source_fs <= 0:
            raise RuntimeError(f"{stem}: invalid sampling frequency {source_fs}")

        # Replace isolated non-finite values before filtering/resampling.
        if not np.isfinite(signal).all():
            signal = signal.copy()
            for lead in range(signal.shape[1]):
                x = signal[:, lead]
                good = np.isfinite(x)
                if not good.any():
                    x[:] = 0.0
                elif not good.all():
                    bad_idx = np.flatnonzero(~good)
                    good_idx = np.flatnonzero(good)
                    x[bad_idx] = np.interp(bad_idx, good_idx, x[good_idx])
                signal[:, lead] = x

        if source_fs != self.target_fs:
            gcd = math.gcd(source_fs, self.target_fs)
            up = self.target_fs // gcd
            down = source_fs // gcd
            signal = resample_poly(
                signal,
                up=up,
                down=down,
                axis=0,
                padtype="line",
            ).astype(np.float32, copy=False)

        # PhysioNet records are 10 s; force identical target length.
        expected_len = int(round(10.0 * self.target_fs))
        signal = self._fix_length(signal, expected_len)
        return signal.astype(np.float32, copy=False)

    def _description_from_row(self, row) -> str:
        parts = ["Patient information"]
        age = row.age
        if pd.notna(age):
            age_value = float(age)
            if age_value.is_integer():
                age_text = str(int(age_value))
            else:
                age_text = f"{age_value:.1f}"
            parts.append(f"age {age_text}")

        sex = str(row.sex).strip()
        if sex and sex.lower() not in {"nan", "unknown", "none", "na", "n/a"}:
            parts.append(sex.lower())

        # IMPORTANT: never put rhythm/diagnosis in the prompt text.
        if len(parts) == 1:
            return "Patient information is unavailable."
        return ", ".join(parts) + "."

    def _build_split_cache(
        self, root: Path, split_df: pd.DataFrame, cache_paths: Dict[str, Path]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        n = len(split_df)
        target_len = int(round(10.0 * self.target_fs))

        tmp_data = cache_paths["data"].with_suffix(".tmp.npy")
        if tmp_data.exists():
            tmp_data.unlink()

        mmap = np.lib.format.open_memmap(
            tmp_data,
            mode="w+",
            dtype=np.float32,
            shape=(n, target_len, 12),
        )

        labels = np.empty((n,), dtype=np.int64)
        descriptions: List[str] = []

        try:
            for i, row in enumerate(
                tqdm(split_df.itertuples(index=False), total=n, desc="Caching ECGs")
            ):
                mmap[i] = self._load_record(root, row.record)
                labels[i] = int(row.label)
                descriptions.append(self._description_from_row(row))
            mmap.flush()
        finally:
            del mmap

        tmp_data.replace(cache_paths["data"])
        np.save(cache_paths["labels"], labels)
        with open(cache_paths["descriptions"], "w", encoding="utf-8") as f:
            json.dump(descriptions, f, ensure_ascii=False)

        data = np.load(cache_paths["data"], mmap_mode="r")
        return data, labels, descriptions

    def get_data(self, split=None):
        split = split or self.split
        root = self._resolve_root()
        manifest = self._load_manifest(root)

        split_df = manifest.loc[manifest["split"] == split].copy()
        if split_df.empty:
            raise RuntimeError(f"No Chapman samples assigned to split={split!r}")
        split_df = split_df.reset_index(drop=True)

        cache_paths = self._cache_paths(root, split)
        if all(path.exists() for path in cache_paths.values()):
            data = np.load(cache_paths["data"], mmap_mode="r")
            labels = np.load(cache_paths["labels"])
            with open(cache_paths["descriptions"], "r", encoding="utf-8") as f:
                descriptions = json.load(f)

            if (
                len(data) != len(split_df)
                or len(labels) != len(split_df)
                or len(descriptions) != len(split_df)
            ):
                raise RuntimeError(
                    "Chapman cache length does not match the split manifest. "
                    "Delete the medtsllm_cache directory and rerun."
                )
            return {
                "data": data,
                "labels": labels.astype(np.int64),
                "descriptions": descriptions,
            }

        print(
            f"[Chapman] building {split} cache: {len(split_df):,} records, "
            f"{self.target_fs} Hz, classes={self.class_names}"
        )
        data, labels, descriptions = self._build_split_cache(
            root, split_df, cache_paths
        )
        return {
            "data": data,
            "labels": labels.astype(np.int64),
            "descriptions": descriptions,
        }

    def map_label(self, label):
        idx = int(label)
        if idx < 0 or idx >= len(self.class_names):
            return str(label)
        return self.class_names[idx]


chapman_datasets = {
    "classification": ChapmanClassificationDataset,
}
