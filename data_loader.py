# -*- coding: utf-8 -*-
"""
Data loader for gene expression prediction
Loads spatial transcriptomics data with gene masking
"""
import os
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import get_fixed_target_genes, IMAGE_ROOT, DATA_ROOT, DATASET_CONFIG


class GeneExpressionDataset(Dataset):
    """
    Dataset for gene expression prediction.

    For each spot:
    - Input: histology image
    - Output: target genes expression (to be predicted)
    """

    def __init__(self, samples, dataset_name, mask_ratio, transform=None, hvg_k=2000):
        """
        Args:
            samples: list of sample dictionaries loaded from .npy files
            dataset_name: "BC", "HER2", or "Kidney"
            mask_ratio: 0.05, 0.15, or 0.30
            transform: image transformations
            hvg_k: number of HVGs used in cell deconv (determines which cell_type_counts key to read)
        """
        self.samples = samples
        self.dataset_name = dataset_name
        self.mask_ratio = mask_ratio
        self.transform = transform
        self.hvg_k = hvg_k

        # Determine cell_type_counts key based on hvg_k
        if hvg_k == 2000:
            self.cell_counts_key = 'cell_type_counts'  # backward compatible
        else:
            self.cell_counts_key = f'cell_type_counts_hvg{hvg_k}'

        # Get fixed gene indices
        self.target_indices, _ = get_fixed_target_genes(dataset_name, mask_ratio)
        self.n_target_genes = len(self.target_indices)
        self.n_cell_types = DATASET_CONFIG[dataset_name]['n_cell_types']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id = sample['sample_id']
        patch_id = sample['patch_id']

        # Load image
        image_path = os.path.join(IMAGE_ROOT, sample_id, f"{patch_id}.png")
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Get expression data (already normalized)
        full_expression = sample['expression_norm']  # shape: (n_genes,)

        # Extract target genes
        target_genes = full_expression[self.target_indices]

        # Get cell type composition for retrieval filtering
        cell_type_counts = sample.get(self.cell_counts_key, None)
        if cell_type_counts is None:
            # fallback to default key, then zeros
            cell_type_counts = sample.get('cell_type_counts', None)
        if cell_type_counts is None:
            cell_type_counts = np.zeros(self.n_cell_types)

        # Get patient_id (added by add_patient_id.py)
        patient_id = sample.get('patient_id', sample_id)  # fallback to sample_id

        return {
            'image': image,
            'target_genes': torch.tensor(target_genes, dtype=torch.float32),
            'cell_composition': torch.tensor(cell_type_counts, dtype=torch.float32),
            'sample_id': sample_id,
            'patch_id': patch_id,
            'patient_id': patient_id,
        }


def load_samples_from_npy(npy_file_path):
    """Load samples from a single .npy file"""
    return list(np.load(npy_file_path, allow_pickle=True))


def get_transforms(is_train=True):
    """Get image transforms"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def create_dataloaders(
    train_npy_files,
    val_npy_files,
    dataset_name,
    mask_ratio,
    batch_size=64,
    num_workers=4,
    cls_val_npy_files=None,
    hvg_k=2000,
):
    """
    Create data loaders for training, validation, and classifier validation.

    Args:
        train_npy_files: list of training .npy file paths
        val_npy_files: list of validation .npy file paths
        dataset_name: "BC", "HER2", or "Kidney"
        batch_size: batch size
        num_workers: number of data loading workers
        cls_val_npy_files: list of classifier-validation .npy file paths (held-out for Stage 3)

    Returns:
        train_loader: training data loader
        val_loaders_by_slide: dict of {slide_name: val_loader}
        train_samples: list of training sample dicts
        cls_val_loaders_by_slide: dict of {slide_name: loader} for cls_val (or None)
        cls_val_samples: list of cls_val sample dicts (or None)
    """
    # Load training samples
    train_samples = []
    for fpath in train_npy_files:
        samples = load_samples_from_npy(fpath)
        train_samples.extend(samples)

    # Create training dataset and loader
    train_dataset = GeneExpressionDataset(
        samples=train_samples,
        dataset_name=dataset_name,
        mask_ratio=mask_ratio,
        transform=get_transforms(is_train=True),
        hvg_k=hvg_k,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Load validation data (separate loader per slide for slide-wise evaluation)
    val_loaders_by_slide = {}
    for fpath in val_npy_files:
        slide_name = Path(fpath).stem  # "SPA100.npy" → "SPA100"
        samples = load_samples_from_npy(fpath)

        val_dataset = GeneExpressionDataset(
            samples=samples,
            dataset_name=dataset_name,
            mask_ratio=mask_ratio,
            transform=get_transforms(is_train=False),
            hvg_k=hvg_k,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        val_loaders_by_slide[slide_name] = val_loader

    # Load classifier-validation data (held-out from training, used for Stage 3 labels)
    cls_val_loaders_by_slide = None
    cls_val_samples = None
    if cls_val_npy_files:
        cls_val_loaders_by_slide = {}
        cls_val_samples = []
        for fpath in cls_val_npy_files:
            slide_name = Path(fpath).stem
            samples = load_samples_from_npy(fpath)
            cls_val_samples.extend(samples)

            cv_dataset = GeneExpressionDataset(
                samples=samples,
                dataset_name=dataset_name,
                mask_ratio=mask_ratio,
                transform=get_transforms(is_train=False),
                hvg_k=hvg_k,
            )
            cv_loader = DataLoader(
                cv_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            cls_val_loaders_by_slide[slide_name] = cv_loader

    print(f"[DataLoader] Dataset: {dataset_name}, Mask: {mask_ratio*100:.0f}%")
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Validation slides: {len(val_loaders_by_slide)}")
    if cls_val_loaders_by_slide:
        print(f"  ClsVal slides: {len(cls_val_loaders_by_slide)} ({len(cls_val_samples)} samples)")
    print(f"  Target genes: {train_dataset.n_target_genes}")

    return train_loader, val_loaders_by_slide, train_samples, cls_val_loaders_by_slide, cls_val_samples


if __name__ == "__main__":
    # Test the data loader
    from cross_fold import get_folds

    dataset_name = "BC"
    mask_ratio = 0.05

    folds = get_folds(dataset_name)
    fold = folds[0]

    train_files = [os.path.join(DATA_ROOT, dataset_name, f) for f in fold['train']]
    val_files = [os.path.join(DATA_ROOT, dataset_name, f) for f in fold['val']]

    train_loader, val_loaders, _, _, _ = create_dataloaders(
        train_npy_files=train_files,
        val_npy_files=val_files,
        dataset_name=dataset_name,
        mask_ratio=mask_ratio,
        batch_size=32
    )

    # Test one batch
    for batch in train_loader:
        print(f"\nBatch test:")
        print(f"  Images: {batch['image'].shape}")
        print(f"  Target genes: {batch['target_genes'].shape}")
        break
