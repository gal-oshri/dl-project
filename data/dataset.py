"""
Dataset loading utilities for CCM training and evaluation.
"""

import logging
from typing import Tuple

from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader

from .collator import CompressionDataCollator

logger = logging.getLogger(__name__)


def load_dialogsum(dataset_name: str = "knkarthick/dialogsum") -> DatasetDict:
    """Load the DialogSum dataset from Hugging Face."""
    dataset = load_dataset(dataset_name)
    logger.info(
        "Dataset loaded — train: %d, validation: %d, test: %d",
        len(dataset["train"]),
        len(dataset["validation"]),
        len(dataset["test"]),
    )
    return dataset


def create_dataloaders(
    dataset: DatasetDict,
    collator: CompressionDataCollator,
    batch_size: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test splits.

    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader).
    """
    train_dl = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_dl = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    test_dl = DataLoader(
        dataset["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    return train_dl, val_dl, test_dl
