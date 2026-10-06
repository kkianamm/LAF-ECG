#!/usr/bin/env python3
"""Install Chapman support into an existing medtsllm-moment checkout.

Run from the repository root:
    python scripts/install_chapman_support.py

The script:
  1) registers PTB-XL/MOMENT and Chapman/MOMENT datasets if missing,
  2) registers ClassificationTask if missing,
  3) registers MomentMedTsLLM if missing,
  4) adds decoder-only MedTsLLM classification support only if it is absent.

Before every edit it writes <file>.pre_chapman_backup once.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil


def backup(path: Path) -> None:
    dst = path.with_name(path.name + ".pre_chapman_backup")
    if not dst.exists():
        shutil.copy2(path, dst)
        print(f"backup: {dst}")


def write_if_changed(path: Path, old: str, new: str) -> None:
    if new == old:
        print(f"unchanged: {path}")
        return
    backup(path)
    path.write_text(new, encoding="utf-8")
    print(f"updated: {path}")


def insert_before(text: str, marker: str, insertion: str) -> str:
    if insertion.strip() in text:
        return text
    pos = text.find(marker)
    if pos < 0:
        raise RuntimeError(f"Could not find insertion marker: {marker!r}")
    return text[:pos] + insertion + text[pos:]


def register_datasets() -> None:
    path = Path("datasets/__init__.py")
    text = path.read_text(encoding="utf-8")
    new = text

    wanted_imports = [
        ("from .ptbxl import ptbxl_datasets", "ptbxl_datasets"),
        ("from .ptbxl_moment import ptbxl_moment_datasets", "ptbxl_moment_datasets"),
        ("from .chapman import chapman_datasets", "chapman_datasets"),
        ("from .chapman_moment import chapman_moment_datasets", "chapman_moment_datasets"),
    ]
    missing_imports = [
        line for line, _ in wanted_imports if line not in new
    ]
    if missing_imports:
        marker = "from .util import"
        block = "\n".join(missing_imports) + "\n\n"
        new = insert_before(new, marker, block)

    wanted_entries = [
        ('"PTB-XL"', '    "PTB-XL": ptbxl_datasets,\n'),
        ('"PTB-XL-MOMENT"', '    "PTB-XL-MOMENT": ptbxl_moment_datasets,\n'),
        ('"CHAPMAN"', '    "CHAPMAN": chapman_datasets,\n'),
        ('"CHAPMAN-MOMENT"', '    "CHAPMAN-MOMENT": chapman_moment_datasets,\n'),
    ]
    missing_entries = [
        line for key, line in wanted_entries if key not in new
    ]
    if missing_entries:
        marker = "\n}\n"
        new = insert_before(new, marker, "".join(missing_entries))

    write_if_changed(path, text, new)

def register_task() -> None:
    path = Path("tasks/__init__.py")
    text = path.read_text(encoding="utf-8")
    new = text

    if "from .classification import ClassificationTask" not in new:
        marker = "from .pretraining import"
        new = insert_before(
            new,
            marker,
            "from .classification import ClassificationTask\n",
        )

    if '"classification": ClassificationTask' not in new:
        # Restrict insertion to task_lookup by inserting before pretraining entry.
        marker = '    "pretraining":'
        new = insert_before(
            new,
            marker,
            '    "classification": ClassificationTask,\n',
        )

    write_if_changed(path, text, new)


def register_model() -> None:
    path = Path("models/__init__.py")
    text = path.read_text(encoding="utf-8")
    new = text

    if "from .moment_medtsllm import MomentMedTsLLM" not in new:
        marker = "from .gpt4ts import"
        new = insert_before(
            new,
            marker,
            "from .moment_medtsllm import MomentMedTsLLM\n",
        )

    if '"moment_medtsllm": MomentMedTsLLM' not in new:
        marker = '"gpt4ts": GPT4TS'
        pos = new.find(marker)
        if pos < 0:
            raise RuntimeError("Could not find gpt4ts model entry.")
        line_start = new.rfind("\n", 0, pos) + 1
        new = (
            new[:line_start]
            + '    "moment_medtsllm": MomentMedTsLLM,\n'
            + new[line_start:]
        )

    write_if_changed(path, text, new)


def patch_decoder_classification() -> None:
    path = Path("models/medtsllm.py")
    text = path.read_text(encoding="utf-8")

    # If a local PTB-XL baseline already added sequence-classification support,
    # leave this file completely untouched.
    supported_line = re.search(r"supported_tasks\s*=\s*\[(.*?)\]", text)
    if supported_line and '"classification"' in supported_line.group(0):
        print("decoder classification: already present")
        return

    new = text

    # 1. Advertise classification.
    new, n = re.subn(
        r'(supported_tasks\s*=\s*\[[^\]]*)(\])',
        lambda m: m.group(1).rstrip() + ', "classification"' + m.group(2),
        new,
        count=1,
    )
    if n != 1:
        raise RuntimeError("Could not update MedTsLLM.supported_tasks")

    # 2. Define K logits for a whole sequence.
    old = (
        '        elif self.task == "segmentation":\n'
        '            self.n_outputs_per_step = 1\n'
        '            assert self.config.tasks.segmentation.mode in '
        '["boundary-prediction", "steps-to-boundary"]\n'
        '        else:\n'
    )
    replacement = (
        '        elif self.task == "segmentation":\n'
        '            self.n_outputs_per_step = 1\n'
        '            assert self.config.tasks.segmentation.mode in '
        '["boundary-prediction", "steps-to-boundary"]\n'
        '        elif self.task == "classification":\n'
        '            self.n_outputs_per_step = self.n_classes\n'
        '        else:\n'
    )
    if old not in new:
        raise RuntimeError("Could not locate MedTsLLM output-task branch")
    new = new.replace(old, replacement, 1)

    old = "        self.n_outputs = self.n_outputs_per_step * self.pred_len\n"
    replacement = (
        old
        + '        if self.task == "classification":\n'
        + "            self.n_outputs = self.n_classes\n"
    )
    if old not in new:
        raise RuntimeError("Could not locate MedTsLLM n_outputs assignment")
    new = new.replace(old, replacement, 1)

    # 3. Return the sequence-level K logits before forecast-style reshaping.
    anchor = "        dec_out = self.output_projection(dec_out)"
    pos = new.find(anchor)
    if pos < 0:
        raise RuntimeError("Could not locate MedTsLLM output projection")
    line_end = new.find("\n", pos)
    addition = (
        '\n        if self.task == "classification":\n'
        '            if self.covariate_mode in ["independent", "merge-end"]:\n'
        '                raise NotImplementedError(\n'
        '                    "Use concat/add/weighted-average/interleave for classification."\n'
        '                )\n'
        '            return dec_out\n'
    )
    new = new[:line_end] + addition + new[line_end:]

    # 4. Supply a classification task prompt if a dataset has no custom prompt.
    prompt_else = (
        '        elif self.task == "segmentation":\n'
        '            self.task_description = '
        'f"Identify the change points in the past {self.seq_len} steps of data '
        'to segment the sequence."\n'
        '        else:\n'
    )
    prompt_replacement = (
        '        elif self.task == "segmentation":\n'
        '            self.task_description = '
        'f"Identify the change points in the past {self.seq_len} steps of data '
        'to segment the sequence."\n'
        '        elif self.task == "classification":\n'
        '            self.task_description = (\n'
        '                f"Classify the entire sequence of {self.seq_len} steps "\n'
        '                "into one diagnostic class."\n'
        '            )\n'
        '        else:\n'
    )
    if prompt_else not in new:
        raise RuntimeError("Could not locate MedTsLLM task-description branch")
    new = new.replace(prompt_else, prompt_replacement, 1)

    write_if_changed(path, text, new)
    print("decoder classification: added")


def main() -> None:
    required = [
        Path("datasets/chapman.py"),
        Path("datasets/chapman_moment.py"),
        Path("tasks/classification.py"),
        Path("models/moment_medtsllm.py"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(
            "Run this script from the medtsllm-moment repository root. "
            f"Missing: {missing}"
        )

    register_datasets()
    register_task()
    register_model()
    patch_decoder_classification()
    print("\nChapman support installed.")


if __name__ == "__main__":
    main()
