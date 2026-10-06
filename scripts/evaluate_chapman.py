#!/usr/bin/env python3
"""
Optional post-training report for a saved classification run.

Usage:
    python scripts/evaluate_chapman.py RUN_ID
    python scripts/evaluate_chapman.py RUN_ID --split test --ckpt best

It uses the repository's ClassificationTask.predict() output and writes:
  * aggregate_metrics.json
  * per_class_metrics.csv
  * confusion_matrix.csv
  * predictions.csv
inside outputs/logs/<RUN_ID>/chapman_evaluation_<split>/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from tasks.classification import ClassificationTask


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--ckpt", default="best")
    args = parser.parse_args()

    task = ClassificationTask.from_run_id(args.run_id, ckpt=args.ckpt)
    loader = getattr(task, f"{args.split}_dataloader")
    output = task.predict(loader)

    # The repository predict() may return (scores, targets) or a dictionary,
    # depending on the exact local revision. Handle both common forms.
    if isinstance(output, dict):
        scores = output.get("scores", output.get("logits"))
        targets = output.get("targets", output.get("labels"))
    elif isinstance(output, (list, tuple)) and len(output) >= 2:
        scores, targets = output[0], output[1]
    else:
        raise RuntimeError(
            "Unsupported ClassificationTask.predict() return type. "
            "Inspect your local tasks/classification.py and adapt this two-line unpack."
        )

    scores = _to_numpy(scores)
    targets = _to_numpy(targets).reshape(-1).astype(int)

    if scores.ndim == 1:
        preds = scores.astype(int)
    else:
        preds = scores.argmax(axis=-1).astype(int)

    dataset = loader.dataset
    class_names = list(
        getattr(dataset, "class_names", [str(i) for i in range(scores.shape[-1])])
    )

    metrics = {
        "accuracy": float(accuracy_score(targets, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds)),
        "f1_macro": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "precision_macro": float(
            precision_score(targets, preds, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(targets, preds, average="macro", zero_division=0)
        ),
    }

    report = classification_report(
        targets,
        preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    per_class = pd.DataFrame(report).T
    per_class = per_class.loc[
        [name for name in class_names if name in per_class.index]
    ]

    cm = confusion_matrix(
        targets, preds, labels=list(range(len(class_names)))
    )
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    pred_df = pd.DataFrame(
        {
            "target_index": targets,
            "target": [class_names[i] for i in targets],
            "pred_index": preds,
            "prediction": [class_names[i] for i in preds],
        }
    )

    out_dir = (
        Path("outputs")
        / "logs"
        / args.run_id
        / f"chapman_evaluation_{args.split}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "aggregate_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    per_class.to_csv(out_dir / "per_class_metrics.csv")
    cm_df.to_csv(out_dir / "confusion_matrix.csv")
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved detailed report to: {out_dir}")


if __name__ == "__main__":
    main()
