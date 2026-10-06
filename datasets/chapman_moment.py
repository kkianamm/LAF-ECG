"""MOMENT wrapper for the Chapman/Shaoxing rhythm dataset.

The Chapman dataset supplies the ECG records/labels/metadata.  The PTB-XL
MOMENT wrapper supplies the exact augmentation, 512-sample high-resolution
windowing, and MedTsLLM x_enc construction used in the existing PTB-XL run.
"""

from .chapman import ChapmanClassificationDataset
from .ptbxl_moment import PTBXLMomentClassificationDataset


class ChapmanMomentClassificationDataset(
    ChapmanClassificationDataset,
    PTBXLMomentClassificationDataset,
):
    """Chapman data with the identical MOMENT input path used for PTB-XL."""

    supported_tasks = ["classification"]


chapman_moment_datasets = {
    "classification": ChapmanMomentClassificationDataset,
}
