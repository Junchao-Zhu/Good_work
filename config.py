# -*- coding: utf-8 -*-
"""
Configuration for Dual-Branch Gene Expression Prediction Model
"""
import os
import numpy as np

# ============================================================
# Data Paths
# ============================================================
DATA_ROOT = "./data"
IMAGE_ROOT = "./data/patches"

# ============================================================
# Dataset Configuration
# ============================================================
DATASET_CONFIG = {
    "BC": {
        "n_genes": 926,
        "n_cell_types": 67,
        "tissue_type": "Breast Cancer"
    },
    "HER2": {
        "n_genes": 967,
        "n_cell_types": 67,
        "tissue_type": "HER2+ Breast Cancer"
    },
    "Kidney": {
        "n_genes": 907,
        "n_cell_types": 88,
        "tissue_type": "Kidney"
    }
}

# Mask ratios for gene prediction
MASK_RATIOS = [0.05, 0.15, 0.30]
MASK_SEED = 42

# ============================================================
# Model Configuration
# ============================================================
BACKBONE_CONFIG = {
    "conch": {
        "name": "conch_vit_b",
        "feature_dim": 512,
        "input_size": 224,
    },
    "densenet121": {
        "name": "densenet121",
        "feature_dim": 1024,
        "input_size": 224,
    },
    "resnet50": {
        "name": "resnet50",
        "feature_dim": 2048,
        "input_size": 224,
    },
    "vit_b": {
        "name": "vit_base_patch16_224",
        "feature_dim": 768,
        "input_size": 224,
    }
}

# ============================================================
# Training Configuration
# ============================================================
TRAIN_CONFIG = {
    "batch_size": 64,
    "lr_encoder": 1e-4,       # Learning rate for backbone + regression head
    "lr_retrieval": 1e-3,     # Learning rate for adapter + gene encoder
    "weight_decay": 1e-4,
    "max_epochs": 100,
    "patience": 15,

    # Retrieval settings
    "projection_dim": 256,
    "temperature": 1.0,
    "top_k": 5,
}


# ============================================================
# Gene Selection Functions
# ============================================================

def get_fixed_target_genes(dataset_name, mask_ratio):
    """
    Get fixed indices of genes to predict (target genes).
    Genes are sorted by HVG (high to low), select top-k as targets.

    Args:
        dataset_name: "BC", "HER2", or "Kidney"
        mask_ratio: ratio of genes to predict (0.05, 0.15, 0.30)

    Returns:
        target_indices: indices of genes to predict
        input_indices: indices of remaining genes (not used in this task)
    """
    n_genes = DATASET_CONFIG[dataset_name]["n_genes"]

    if mask_ratio >= 1:
        n_target = int(mask_ratio)
    else:
        n_target = int(n_genes * mask_ratio)

    n_target = min(n_target, n_genes)

    # Top-k genes as targets (indices 0 to n_target-1)
    target_indices = np.arange(n_target)
    input_indices = np.arange(n_target, n_genes)

    return target_indices, input_indices


def get_model_dims(dataset_name, mask_ratio):
    """
    Get output dimension for the model.

    Args:
        dataset_name: "BC", "HER2", or "Kidney"
        mask_ratio: ratio of genes to predict

    Returns:
        input_dim: number of input genes (not used)
        output_dim: number of genes to predict
    """
    target_indices, input_indices = get_fixed_target_genes(dataset_name, mask_ratio)
    return len(input_indices), len(target_indices)


if __name__ == "__main__":
    print("=" * 70)
    print("Configuration Summary")
    print("=" * 70)

    for dataset in ["BC", "HER2", "Kidney"]:
        print(f"\nDataset: {dataset}")
        print(f"  Total genes: {DATASET_CONFIG[dataset]['n_genes']}")

        for ratio in MASK_RATIOS:
            _, output_dim = get_model_dims(dataset, ratio)
            print(f"  Mask {ratio*100:.0f}%: predict {output_dim} genes")
