# -*- coding: utf-8 -*-
"""
Evaluation metrics for gene expression prediction
Consistent with baselines (tensor-based PCC)
"""
import torch
import numpy as np
from scipy.stats import pearsonr


def calculate_pcc(pred, target):
    """
    Calculate per-gene Pearson Correlation Coefficient (tensor version)
    Same as baselines for fair comparison.

    Args:
        pred: (n_samples, n_genes) predictions
        target: (n_samples, n_genes) ground truth

    Returns:
        mean PCC across all genes
    """
    target = target.float()
    pred = pred.float()

    # Center the data
    x = pred - pred.mean(dim=0, keepdim=True)
    y = target - target.mean(dim=0, keepdim=True)

    # Compute covariance and variances
    covariance = (x * y).sum(dim=0)
    variance_x = (x ** 2).sum(dim=0)
    variance_y = (y ** 2).sum(dim=0)

    # Compute PCC per gene
    pcc = covariance / torch.sqrt(variance_x * variance_y + 1e-8)

    return pcc.mean()


def calculate_pcc_numpy(pred, target):
    """
    Calculate per-gene PCC using numpy/scipy (for verification)

    Args:
        pred: (n_samples, n_genes) predictions
        target: (n_samples, n_genes) ground truth

    Returns:
        mean PCC across genes
    """
    pcc_list = []
    for j in range(pred.shape[1]):
        if target[:, j].std() > 1e-6 and pred[:, j].std() > 1e-6:
            corr, _ = pearsonr(pred[:, j], target[:, j])
            if not np.isnan(corr):
                pcc_list.append(corr)
    return np.mean(pcc_list) if pcc_list else 0.0


def compute_metrics(pred, target):
    """
    Compute all evaluation metrics.

    Args:
        pred: predictions (numpy array or tensor)
        target: ground truth (numpy array or tensor)

    Returns:
        dict with mse, mae, pcc
    """
    if isinstance(pred, torch.Tensor):
        pred_np = pred.numpy()
        target_np = target.numpy()
        pred_t = pred
        target_t = target
    else:
        pred_np = pred
        target_np = target
        pred_t = torch.tensor(pred)
        target_t = torch.tensor(target)

    mse = np.mean((pred_np - target_np) ** 2)
    mae = np.mean(np.abs(pred_np - target_np))
    pcc = calculate_pcc(pred_t, target_t).item()

    return {'mse': mse, 'mae': mae, 'pcc': pcc}
