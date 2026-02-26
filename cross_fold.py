# -*- coding: utf-8 -*-
"""
Cross-fold validation splits for gene expression prediction
Consistent with baselines for fair comparison
"""
from sklearn.model_selection import KFold
import random


def make_range(prefix, start, end):
    """Generate slide IDs in range"""
    step = -1 if start >= end else 1
    return [f"{prefix}{i}.npy" for i in range(start, end + step, step)]


def select_cls_val_her2(train_slides):
    """Select 1 patient (3 slides) from training set for classifier validation.

    Patients E-H each have 3 slides. Pick the first one whose slides are all in train.
    """
    # HER2 patient groups (3-slide patients only, to minimize training data loss)
    patients_3slide = {
        'E': ['SPA128.npy', 'SPA129.npy', 'SPA130.npy'],
        'F': ['SPA125.npy', 'SPA126.npy', 'SPA127.npy'],
        'G': ['SPA122.npy', 'SPA123.npy', 'SPA124.npy'],
        'H': ['SPA119.npy', 'SPA120.npy', 'SPA121.npy'],
    }
    train_set = set(train_slides)
    for pid in ['E', 'F', 'G', 'H']:
        slides = patients_3slide[pid]
        if all(s in train_set for s in slides):
            return slides
    # Fallback: pick the last 3 slides in train
    return train_slides[-3:]


def select_cls_val_kidney(train_slides, n_cls_val=2):
    """Select n_cls_val slides from training set for classifier validation (Kidney)."""
    return train_slides[-n_cls_val:]


def select_cls_val_bc(train_slides, n_cls_val=6):
    """Select n_cls_val slides from training set for classifier validation (BC).

    With 65 total slides, taking 6 for cls_val gives ~2700 spots for
    classifier training while only reducing training by ~11%.
    """
    return train_slides[-n_cls_val:]


def generate_her2_splits():
    """Generate 4-fold splits for HER2 dataset (36 slides)"""
    her2_all = make_range("SPA", 154, 119)

    # First 24 samples: 6 per group (6×4)
    her2_6x_samples = [her2_all[i * 6:(i + 1) * 6] for i in range(4)]
    # Last 12 samples: 3 per group (3×4)
    her2_3x_samples = [her2_all[24 + i * 3:24 + (i + 1) * 3] for i in range(4)]

    folds = []
    for i in range(4):
        val = her2_6x_samples[i] + her2_3x_samples[i]
        train = [sid for sid in her2_all if sid not in val]
        cls_val = select_cls_val_her2(train)
        train = [sid for sid in train if sid not in cls_val]
        folds.append({'train': train, 'val': val, 'cls_val': cls_val})
    return folds


def generate_bc_splits():
    """Generate 4-fold splits for BC dataset (65 slides, excluding SPA111, SPA112)"""
    bc_all = [s for s in make_range("SPA", 118, 51) if s not in ("SPA112.npy", "SPA111.npy")]
    bc_samples = [bc_all[i * 3:(i + 1) * 3] for i in range(len(bc_all) // 3)]
    random.seed(42)
    random.shuffle(bc_samples)

    folds = []
    kf = KFold(n_splits=4)
    for train_idx, val_idx in kf.split(bc_samples):
        train_samples = [bc_samples[i] for i in train_idx]
        val_samples = [bc_samples[i] for i in val_idx]
        val_first_slide = [sample[0] for sample in val_samples]
        # Select cls_val from pure training slides only (not val_first_slide)
        pure_train = sum(train_samples, [])
        cls_val = select_cls_val_bc(pure_train)
        # Build final train: pure_train - cls_val + val_first_slide
        train = [sid for sid in pure_train if sid not in cls_val] + val_first_slide
        val = sum(val_samples, [])
        folds.append({'train': train, 'val': val, 'cls_val': cls_val})
    return folds


def generate_kidney_splits():
    """Generate 4-fold splits for Kidney dataset (23 slides)"""
    kidney_all = make_range("NCBI", 714, 692)
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    folds = []
    for train_idx, val_idx in kf.split(kidney_all):
        train = [kidney_all[i] for i in train_idx]
        val = [kidney_all[i] for i in val_idx]
        cls_val = select_cls_val_kidney(train)
        train = [sid for sid in train if sid not in cls_val]
        folds.append({'train': train, 'val': val, 'cls_val': cls_val})
    return folds


def get_folds(dataset_name):
    """Get 4-fold splits for the specified dataset"""
    if dataset_name == "HER2":
        return generate_her2_splits()
    elif dataset_name == "BC":
        return generate_bc_splits()
    elif dataset_name == "Kidney":
        return generate_kidney_splits()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


if __name__ == "__main__":
    for dataset in ["BC", "HER2", "Kidney"]:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print("="*60)
        folds = get_folds(dataset)
        for i, fold in enumerate(folds):
            cls_val = fold.get('cls_val', [])
            print(f"Fold {i}: Train={len(fold['train'])} slides, Val={len(fold['val'])} slides, "
                  f"ClsVal={len(cls_val)} slides {cls_val}")
