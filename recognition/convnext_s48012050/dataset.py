"""Contains the data loader for loading and preprocessing the data, 
plus additional data manipulations
"""

import copy
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CLASS_TO_IDX = {"AD": 0, "NC": 1}
IMAGE_SIZE = (240, 256)  # (height, width)


def make_transforms() -> transforms.Compose:
    """
    Create the transformation composition used for the dataset.
    Currently just scales pixel values, but this method can be adapted
    later to perform augmentation

    Returns:
        transforms.Compose: The transformations
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])


def validation_split(dataset, val_fraction: float, seed: int | None = None) -> Tuple[list, list]:
    """
    Split indices within each class, preserving class representation and eforcing patient level separation

    Args:
        dataset (_type_): The dataset to split for a given class
        val_fraction (float): The fraction of data to use for validation (appox since we maintain patient level separation)
        seed (int | None): Seed for reproducibility. If None, a random seed is used.

    Returns:
        Tuple[list, list]: (train_indices, val_indices) in that order
    """
    generator = torch.Generator().manual_seed(seed if seed is not None else torch.seed())
    patient_to_indices: dict[str, list[int]] = {}
    patient_to_class: dict[str, int] = {}

    # map each patient number to the indices of their images in the dataset,
    # and also map each patient to their class
    for index, (path, class_idx) in enumerate(dataset.samples):
        patient_id = _patient_id(path)

        # explicitly ensure the same patient does not appear in multiple class folders
        # useful for the dataset I manually downloaded
        previous_class = patient_to_class.setdefault(patient_id, class_idx)
        if previous_class != class_idx:
            raise ValueError(
                f"Patient {patient_id} appears in multiple class folders"
            )
        
        patient_to_indices.setdefault(patient_id, []).append(index)


    class_to_patients: dict[int, list[str]] = {0: [], 1: []}
    for patient_id, class_idx in patient_to_class.items():
        class_to_patients[class_idx].append(patient_id)

    # now we have patients in each class, we can shuffle them and split into train/val
    train_indices, val_indices = [], []
    for class_idx in sorted(class_to_patients):
        patients = class_to_patients[class_idx]
        permutation = torch.randperm(len(patients), generator=generator).tolist()
        shuffled_patients = [patients[i] for i in permutation]
        n_val = int(round(len(shuffled_patients) * val_fraction))

        val_patients = shuffled_patients[:n_val]
        train_patients = shuffled_patients[n_val:]
        for patient_id in train_patients:
            train_indices.extend(patient_to_indices[patient_id])
        for patient_id in val_patients:
            val_indices.extend(patient_to_indices[patient_id])

    train_indices = [train_indices[i] for i in torch.randperm(len(train_indices), generator=generator).tolist()]
    val_indices = [val_indices[i] for i in torch.randperm(len(val_indices), generator=generator).tolist()]
    return train_indices, val_indices

def _patient_id(path: str | Path) -> str:
    """Extract the patient identifier from a 'patient_slice.ext' filename."""
    stem = Path(path).stem
    patient_id, separator, slice_id = stem.partition("_")
    if not separator or not patient_id or not slice_id:
        raise ValueError(
            f"Image filename does not match '<patient number>_<slice number>.<ext>': {path}"
        )
    return patient_id


def create_dataloaders(
    data_dir: str | Path = "datasets",
    batch_size: int = 32,
    val_fraction: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool | None = None,
) -> Dict[str, DataLoader]:
    """Create train, validation, and test DataLoaders.

    data_dir should contain train/AD, train/NC, test/AD, and test/NC.

    Returns loaders keyed by 'train', 'val', and 'test'. Images have shape (B, 1, 240, 256), are grayscale, and have values in [0, 1].
    Note: If val_fraction == 0, the 'val' key will not be present in the returned dictionary.
    """
    if val_fraction < 0 or val_fraction >= 1:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    root = Path(data_dir)
    train_dir, test_dir = root / "train", root / "test"

    for directory in (train_dir / "AD", train_dir / "NC", test_dir / "AD", test_dir / "NC"):
        if not directory.is_dir():
            raise FileNotFoundError(f"Expected class directory was not found: {directory}")

    # ImageFolder creates the class labels for us alphabetically, so AD=0, NC=1
    train_dataset = datasets.ImageFolder(train_dir, transform=make_transforms())
    val_dataset = copy.deepcopy(train_dataset)  # We will subset this later
    test_dataset = datasets.ImageFolder(test_dir, transform=make_transforms())

    if val_fraction > 0:
        train_indices, val_indices = validation_split(train_dataset, val_fraction, seed)
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
    else:
        train_dataset, val_dataset = train_dataset, None

    # Enable page-locked memory if we have a GPU
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader_generator = torch.Generator().manual_seed(seed)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    loaders: Dict[str, DataLoader] = {
        "train": DataLoader(train_dataset, shuffle=True, generator=loader_generator, **common),
        "test": DataLoader(test_dataset, shuffle=False, **common),
    }

    if val_dataset is not None:
        loaders["val"] = DataLoader(val_dataset, shuffle=False, **common)

    print(f"Class mapping: {CLASS_TO_IDX}")
    for split, loader in loaders.items():
        print(f"{split}: {len(loader.dataset)} images, {len(loader)} batches") # type: ignore
    return loaders
