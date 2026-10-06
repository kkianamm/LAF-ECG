#!/usr/bin/env python3
from pathlib import Path


def require(path: str, text: str):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"MISSING: {p}")
    content = p.read_text(encoding="utf-8")
    if text not in content:
        raise SystemExit(f"NOT REGISTERED: expected {text!r} in {p}")
    print(f"OK: {p} -> {text}")


def main():
    require("datasets/chapman.py", "ChapmanClassificationDataset")
    require("datasets/chapman_moment.py", "ChapmanMomentClassificationDataset")
    require("datasets/__init__.py", '"CHAPMAN"')
    require("datasets/__init__.py", '"CHAPMAN-MOMENT"')
    require("models/__init__.py", '"moment_medtsllm"')
    require("tasks/__init__.py", '"classification"')

    med = Path("models/medtsllm.py").read_text(encoding="utf-8")
    if '"classification"' in med:
        print("OK: decoder-only MedTsLLM classification support is present")
    else:
        print(
            "WARNING: decoder-only MedTsLLM classification support is absent. "
            "Run scripts/install_chapman_support.py before the baseline."
        )

    print("\nChapman/MOMENT integration is registered.")


if __name__ == "__main__":
    main()
