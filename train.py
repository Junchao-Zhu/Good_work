# -*- coding: utf-8 -*-
"""
Training Script for Dual-Backbone Gene Expression Prediction

Stage 1: DenseNet-121 + GeneEncoder    <- BLEEP contrastive (temp=0.07)
Stage 2: ViT-B + RegressionHead        <- MSE loss
Stage 3: BranchClassifier              <- MSE + regularization (both frozen)

Key design choices:
  - BLEEP uses its own DenseNet-121 backbone (end-to-end)
  - Temperature = 0.07 (standard CLIP/BLEEP)
  - Training order: BLEEP first, then regression
  - Evaluates retrieval with AND without cell type filtering
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter

from models import DualBackboneModel, build_gene_embed_database
from data_loader import create_dataloaders
from cross_fold import get_folds
from eval_metric import compute_metrics
from config import DATA_ROOT, DATASET_CONFIG, get_model_dims


# ---------------------------------------------------------------------------
# Stage 1: BLEEP contrastive (DenseNet-121 + GeneEncoder)
# ---------------------------------------------------------------------------

def train_epoch_stage1(model, train_loader, optimizer, device):
    """Stage 1: Train DenseNet-121 + GeneEncoder with BLEEP contrastive loss"""
    model.bleep_encoder.train()
    model.bleep_projection.train()
    model.gene_encoder.train()

    total_loss = 0
    n_batches = 0

    pbar = tqdm(train_loader, desc='Stage1-BLEEP', ncols=120)
    for batch in pbar:
        images = batch['image'].to(device)
        target_genes = batch['target_genes'].to(device)

        # DenseNet forward -> projection -> L2 normalize
        image_embed = model.get_bleep_image_embed(images)
        gene_embed = model.gene_encoder(target_genes)

        optimizer.zero_grad()
        loss = model.compute_contrastive_loss(image_embed, gene_embed)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'Con': f'{total_loss / n_batches:.4f}'})

    return {'loss_con': total_loss / n_batches}


@torch.no_grad()
def evaluate_retrieval(
    model, val_loader, device,
    gene_embeds_db, target_genes_db,
    db_cell_comp=None,
    count_threshold=0.5, comp_sim_threshold=0.5, cell_sim_weight=0.3,
):
    """
    Evaluate retrieval branch on a single slide.
    Returns metrics both with and without cell type filtering.
    """
    model.bleep_encoder.eval()
    model.bleep_projection.eval()

    all_pred_ret_nofilter = []
    all_pred_ret_filtered = []
    all_targets = []

    gene_embeds_db_dev = gene_embeds_db.to(device)
    target_genes_db_dev = target_genes_db.to(device)
    if db_cell_comp is not None:
        db_cell_comp_dev = db_cell_comp.to(device)

    for batch in val_loader:
        images = batch['image'].to(device)
        target_genes = batch['target_genes'].to(device)

        # No filter retrieval
        out_nofilter = model.inference_ret_only(
            images=images,
            gene_embeds_db=gene_embeds_db_dev,
            target_genes_db=target_genes_db_dev,
        )
        all_pred_ret_nofilter.append(out_nofilter['pred_ret'].cpu())

        # With cell type filter (if available)
        if db_cell_comp is not None:
            query_cell_comp = batch['cell_composition'].to(device)
            out_filtered = model.inference_ret_only(
                images=images,
                gene_embeds_db=gene_embeds_db_dev,
                target_genes_db=target_genes_db_dev,
                query_cell_comp=query_cell_comp,
                db_cell_comp=db_cell_comp_dev,
                count_threshold=count_threshold,
                comp_sim_threshold=comp_sim_threshold,
                cell_sim_weight=cell_sim_weight,
            )
            all_pred_ret_filtered.append(out_filtered['pred_ret'].cpu())

        all_targets.append(target_genes.cpu())

    pred_ret_nofilter = torch.cat(all_pred_ret_nofilter, dim=0)
    targets = torch.cat(all_targets, dim=0)

    results = {
        'ret_nofilter': compute_metrics(pred_ret_nofilter, targets),
        'pred_ret_nofilter': pred_ret_nofilter.numpy(),
        'targets': targets.numpy(),
    }

    if all_pred_ret_filtered:
        pred_ret_filtered = torch.cat(all_pred_ret_filtered, dim=0)
        results['ret_filtered'] = compute_metrics(pred_ret_filtered, targets)
        results['pred_ret_filtered'] = pred_ret_filtered.numpy()

    return results


# ---------------------------------------------------------------------------
# Stage 2: Regression (ViT-B + RegressionHead)
# ---------------------------------------------------------------------------

def train_epoch_stage2(model, train_loader, optimizer, device,
                       gene_embeds_db=None, target_genes_db=None,
                       db_cell_comp=None, lambda_ret=0.0,
                       count_threshold=0.5, comp_sim_threshold=0.15, cell_sim_weight=0.3):
    """Stage 2: MSE(pred_reg, GT) + lambda_ret * MSE(pred_reg, pred_ret)"""
    model.reg_encoder.train()
    model.regression_head.train()

    use_ret = lambda_ret > 0 and gene_embeds_db is not None
    if use_ret:
        gene_embeds_db_d = gene_embeds_db.to(device)
        target_genes_db_d = target_genes_db.to(device)
        db_cell_comp_d = db_cell_comp.to(device) if db_cell_comp is not None else None

    total_mse_gt = 0
    total_mse_ret = 0
    n_batches = 0

    pbar = tqdm(train_loader, desc='Stage2-Reg', ncols=120)
    for batch in pbar:
        images = batch['image'].to(device)
        target_genes = batch['target_genes'].to(device)

        reg_out = model.reg_encoder(images)
        pred_reg = model.regression_head(reg_out['features'])

        loss_gt = F.mse_loss(pred_reg, target_genes)

        if use_ret:
            with torch.no_grad():
                image_embed = model.get_bleep_image_embed(images)
                query_cc = batch['cell_composition'].to(device)
                pred_ret, _, _ = model.retrieve(
                    image_embed, gene_embeds_db_d, target_genes_db_d,
                    query_cell_comp=query_cc, db_cell_comp=db_cell_comp_d,
                    count_threshold=count_threshold,
                    comp_sim_threshold=comp_sim_threshold,
                    cell_sim_weight=cell_sim_weight)
            loss_ret = F.mse_loss(pred_reg, pred_ret.detach())
            loss = loss_gt + lambda_ret * loss_ret
            total_mse_ret += loss_ret.item()
        else:
            loss = loss_gt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_mse_gt += loss_gt.item()
        n_batches += 1
        if use_ret:
            pbar.set_postfix({'MSE_gt': f'{total_mse_gt/n_batches:.4f}',
                              'MSE_ret': f'{total_mse_ret/n_batches:.4f}'})
        else:
            pbar.set_postfix({'MSE': f'{total_mse_gt/n_batches:.4f}'})

    result = {'loss_reg': total_mse_gt / n_batches}
    if use_ret:
        result['loss_ret'] = total_mse_ret / n_batches
    return result


@torch.no_grad()
def evaluate_regression(model, val_loader, device):
    """Regression-only evaluation for Stage 2"""
    model.reg_encoder.eval()
    model.regression_head.eval()

    all_preds = []
    all_targets = []

    for batch in val_loader:
        images = batch['image'].to(device)
        target_genes = batch['target_genes'].to(device)

        reg_out = model.reg_encoder(images)
        pred = model.regression_head(reg_out['features'])

        all_preds.append(pred.cpu())
        all_targets.append(target_genes.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(preds, targets)

    return {
        'regression': metrics,
        'pred_reg': preds.numpy(),
        'targets': targets.numpy(),
    }


# ---------------------------------------------------------------------------
# Full evaluation (both branches)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_slide(
    model, val_loader, device,
    gene_embeds_db, target_genes_db, db_cell_comp,
    count_threshold=0.5, comp_sim_threshold=0.5, cell_sim_weight=0.3,
):
    """Evaluate both branches on a single slide"""
    model.eval()

    all_pred_reg = []
    all_pred_ret = []
    all_pred_ret_nofilter = []
    all_targets = []
    all_features = []

    gene_embeds_db_dev = gene_embeds_db.to(device)
    target_genes_db_dev = target_genes_db.to(device)
    db_cell_comp_dev = db_cell_comp.to(device)

    for batch in val_loader:
        images = batch['image'].to(device)
        target_genes = batch['target_genes'].to(device)
        query_cell_comp = batch['cell_composition'].to(device)

        # Full inference (with filter)
        outputs = model.inference(
            images=images,
            gene_embeds_db=gene_embeds_db_dev,
            target_genes_db=target_genes_db_dev,
            query_cell_comp=query_cell_comp,
            db_cell_comp=db_cell_comp_dev,
            count_threshold=count_threshold,
            comp_sim_threshold=comp_sim_threshold,
            cell_sim_weight=cell_sim_weight,
        )

        # Also get no-filter retrieval for comparison
        out_nofilter = model.inference_ret_only(
            images=images,
            gene_embeds_db=gene_embeds_db_dev,
            target_genes_db=target_genes_db_dev,
        )

        all_pred_reg.append(outputs['pred_reg'].cpu())
        all_pred_ret.append(outputs['pred_ret'].cpu())
        all_pred_ret_nofilter.append(out_nofilter['pred_ret'].cpu())
        all_targets.append(target_genes.cpu())
        all_features.append(outputs['features'].cpu())

    pred_reg = torch.cat(all_pred_reg, dim=0)
    pred_ret = torch.cat(all_pred_ret, dim=0)
    pred_ret_nofilter = torch.cat(all_pred_ret_nofilter, dim=0)
    targets = torch.cat(all_targets, dim=0)
    features = torch.cat(all_features, dim=0)

    metrics_reg = compute_metrics(pred_reg, targets)
    metrics_ret = compute_metrics(pred_ret, targets)
    metrics_ret_nf = compute_metrics(pred_ret_nofilter, targets)

    return {
        'regression': metrics_reg,
        'retrieval': metrics_ret,
        'retrieval_nofilter': metrics_ret_nf,
        'pred_reg': pred_reg.numpy(),
        'pred_ret': pred_ret.numpy(),
        'pred_ret_nofilter': pred_ret_nofilter.numpy(),
        'targets': targets.numpy(),
        'features': features,
    }


# ---------------------------------------------------------------------------
# Stage 3: Soft Mixing
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_soft_mixing_data(
    model, cls_val_loaders_by_slide, device,
    gene_embeds_db, all_target_genes, all_cell_comp,
    count_threshold=0.5, comp_sim_threshold=0.15, cell_sim_weight=0.3,
):
    """Generate training data for soft mixing on cls_val slides"""
    model.eval()

    all_features_list = []
    all_pred_reg_list = []
    all_pred_ret_list = []
    all_gt_list = []

    gene_embeds_db_dev = gene_embeds_db.to(device)
    all_target_genes_dev = all_target_genes.to(device)
    all_cell_comp_dev = all_cell_comp.to(device)

    logging.info(f"  Generating soft mixing data on {len(cls_val_loaders_by_slide)} cls_val slides")

    for slide_name, loader in tqdm(cls_val_loaders_by_slide.items(), desc="GenMixData", ncols=120):
        slide_n_total = 0

        for batch in loader:
            images = batch['image'].to(device)
            target_genes = batch['target_genes'].to(device)
            query_cell_comp = batch['cell_composition'].to(device)

            outputs = model.inference(
                images=images,
                gene_embeds_db=gene_embeds_db_dev,
                target_genes_db=all_target_genes_dev,
                query_cell_comp=query_cell_comp,
                db_cell_comp=all_cell_comp_dev,
                count_threshold=count_threshold,
                comp_sim_threshold=comp_sim_threshold,
                cell_sim_weight=cell_sim_weight,
            )

            features = outputs['features'].cpu()
            pred_reg = outputs['pred_reg'].cpu()
            pred_ret = outputs['pred_ret'].cpu()
            gt = target_genes.cpu()

            all_features_list.append(features)
            all_pred_reg_list.append(pred_reg)
            all_pred_ret_list.append(pred_ret)
            all_gt_list.append(gt)

            slide_n_total += len(features)

        slide_reg = torch.cat([r for r in all_pred_reg_list[-len(loader):]], dim=0)
        slide_ret = torch.cat([r for r in all_pred_ret_list[-len(loader):]], dim=0)
        slide_gt = torch.cat([g for g in all_gt_list[-len(loader):]], dim=0)
        reg_mse = ((slide_reg - slide_gt) ** 2).mean().item()
        ret_mse = ((slide_ret - slide_gt) ** 2).mean().item()
        avg_mse = ((0.5 * slide_reg + 0.5 * slide_ret - slide_gt) ** 2).mean().item()
        logging.info(f"    {slide_name}: {slide_n_total} spots, "
                     f"reg_mse={reg_mse:.4f}, ret_mse={ret_mse:.4f}, avg_mse={avg_mse:.4f}")

    all_features = torch.cat(all_features_list, dim=0)
    all_pred_reg = torch.cat(all_pred_reg_list, dim=0)
    all_pred_ret = torch.cat(all_pred_ret_list, dim=0)
    all_gt = torch.cat(all_gt_list, dim=0)

    logging.info(f"  Total: {len(all_features)} spots")

    return all_features, all_pred_reg, all_pred_ret, all_gt


def train_soft_mixing(
    model, features, pred_reg, pred_ret, gt,
    device, n_epochs=100, lr=1e-3, reg_lambda=0.5, reg_type='l2',
    hidden_dim=128, spot_repr_dim=64, dropout=0.2,
):
    """Train soft mixing network with MSE loss + regularization"""
    from models.heads import BranchClassifier

    input_dim = model.branch_classifier.spot_encoder[0].in_features
    model.branch_classifier = BranchClassifier(
        input_dim=input_dim, hidden_dim=hidden_dim,
        spot_repr_dim=spot_repr_dim, dropout=dropout,
    ).to(device)

    model.branch_classifier.train()
    for name, param in model.named_parameters():
        param.requires_grad = 'branch_classifier' in name

    optimizer = AdamW(model.get_optimizer_params_cls(), lr=lr, weight_decay=1e-3)

    features_dev = features.to(device)
    pred_reg_dev = pred_reg.to(device)
    pred_ret_dev = pred_ret.to(device)
    gt_dev = gt.to(device)

    dataset = torch.utils.data.TensorDataset(features_dev, pred_reg_dev, pred_ret_dev, gt_dev)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    best_loss = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        total_mse = 0
        total = 0

        for feat_b, reg_b, ret_b, gt_b in loader:
            delta = model.branch_classifier.get_delta(feat_b)
            alpha = 0.5 + delta
            pred = alpha.unsqueeze(1) * ret_b + (1 - alpha.unsqueeze(1)) * reg_b

            mse_loss = F.mse_loss(pred, gt_b)

            if reg_type == 'l1':
                reg_loss = delta.abs().mean()
            else:
                reg_loss = (delta ** 2).mean()

            loss = mse_loss + reg_lambda * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_mse += mse_loss.item() * len(feat_b)
            total += len(feat_b)

        avg_mse = total_mse / total
        if avg_mse < best_loss:
            best_loss = avg_mse
            best_state = {k: v.clone() for k, v in model.branch_classifier.state_dict().items()}

    model.branch_classifier.load_state_dict(best_state)
    model.branch_classifier.eval()
    for param in model.parameters():
        param.requires_grad = True

    return best_state


def per_gene_alpha(pred_reg, pred_ret, gt, reg_lambda=1.0):
    """Closed-form per-gene alpha with regularization toward 0.5"""
    if isinstance(pred_reg, torch.Tensor):
        pred_reg = pred_reg.numpy()
        pred_ret = pred_ret.numpy()
        gt = gt.numpy()

    r = pred_ret - pred_reg
    g = gt - pred_reg

    numer = (r * g).sum(axis=0) + reg_lambda * 0.5
    denom = (r * r).sum(axis=0) + reg_lambda

    alpha_g = numer / denom
    alpha_g = np.clip(alpha_g, 0.0, 1.0)
    return alpha_g


def _eval_predictions(pred, gt):
    m = compute_metrics(pred, gt)
    return m['mse'], m['mae'], m['pcc']


def run_stage3_all_variants(
    model, cls_features, cls_pred_reg, cls_pred_ret, cls_gt,
    device, save_path,
):
    """Stage 3: Exhaustive search over classifier variants"""
    reg_np = cls_pred_reg.cpu().numpy() if isinstance(cls_pred_reg, torch.Tensor) else cls_pred_reg
    ret_np = cls_pred_ret.cpu().numpy() if isinstance(cls_pred_ret, torch.Tensor) else cls_pred_ret
    gt_np = cls_gt.cpu().numpy() if isinstance(cls_gt, torch.Tensor) else cls_gt

    results = []

    # === Baselines ===
    logging.info(f"\n  --- Baselines ---")

    mse, mae, pcc = _eval_predictions(reg_np, gt_np)
    results.append({'method': 'regression_only', 'type': 'baseline', 'mse': mse, 'mae': mae, 'pcc': pcc, 'alpha_info': 'alpha=0'})
    logging.info(f"    {'regression_only':<30s}  MSE={mse:.6f}  MAE={mae:.6f}  PCC={pcc:.4f}")

    mse, mae, pcc = _eval_predictions(ret_np, gt_np)
    results.append({'method': 'retrieval_only', 'type': 'baseline', 'mse': mse, 'mae': mae, 'pcc': pcc, 'alpha_info': 'alpha=1'})
    logging.info(f"    {'retrieval_only':<30s}  MSE={mse:.6f}  MAE={mae:.6f}  PCC={pcc:.4f}")

    fixed_alphas = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    for alpha_val in fixed_alphas:
        pred = alpha_val * ret_np + (1 - alpha_val) * reg_np
        mse, mae, pcc = _eval_predictions(pred, gt_np)
        name = 'simple_avg' if alpha_val == 0.5 else f'fixed_alpha_{alpha_val}'
        results.append({'method': name, 'type': 'fixed', 'mse': mse, 'mae': mae, 'pcc': pcc, 'alpha_info': f'alpha={alpha_val}'})
        logging.info(f"    {name:<30s}  MSE={mse:.6f}  MAE={mae:.6f}  PCC={pcc:.4f}")

    # Oracle
    reg_mse_spot = ((reg_np - gt_np) ** 2).mean(axis=1)
    ret_mse_spot = ((ret_np - gt_np) ** 2).mean(axis=1)
    oracle_use_ret = ret_mse_spot < reg_mse_spot
    oracle_pred = np.where(oracle_use_ret[:, None], ret_np, reg_np)
    mse, mae, pcc = _eval_predictions(oracle_pred, gt_np)
    results.append({'method': 'oracle_spot', 'type': 'oracle', 'mse': mse, 'mae': mae, 'pcc': pcc,
                    'alpha_info': f'ret={oracle_use_ret.mean()*100:.1f}%'})
    logging.info(f"    {'oracle_spot':<30s}  MSE={mse:.6f}  MAE={mae:.6f}  PCC={pcc:.4f}  (ret={oracle_use_ret.mean()*100:.1f}%)")

    # === Neural variants ===
    logging.info(f"\n  --- Neural variants ---")

    neural_configs = []
    for reg_type in ['l1', 'l2']:
        for lam in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
            neural_configs.append({
                'name': f'{reg_type}_lam{lam}',
                'reg_type': reg_type, 'reg_lambda': lam,
                'hidden_dim': 128, 'spot_repr_dim': 64, 'dropout': 0.2, 'lr': 1e-3,
            })
    for lam in [1.0, 5.0, 20.0]:
        neural_configs.append({
            'name': f'l2_small_lam{lam}',
            'reg_type': 'l2', 'reg_lambda': lam,
            'hidden_dim': 64, 'spot_repr_dim': 32, 'dropout': 0.2, 'lr': 1e-3,
        })
    for lam in [1.0, 5.0, 20.0]:
        neural_configs.append({
            'name': f'l2_drop05_lam{lam}',
            'reg_type': 'l2', 'reg_lambda': lam,
            'hidden_dim': 128, 'spot_repr_dim': 64, 'dropout': 0.5, 'lr': 1e-3,
        })

    best_neural_pcc = -float('inf')
    best_neural_method = None
    best_neural_state = None
    all_neural_states = {}  # Save ALL neural states

    for cfg in neural_configs:
        state = train_soft_mixing(
            model=model,
            features=cls_features,
            pred_reg=cls_pred_reg,
            pred_ret=cls_pred_ret,
            gt=cls_gt,
            device=device,
            n_epochs=100,
            lr=cfg['lr'],
            reg_lambda=cfg['reg_lambda'],
            reg_type=cfg['reg_type'],
            hidden_dim=cfg['hidden_dim'],
            spot_repr_dim=cfg['spot_repr_dim'],
            dropout=cfg['dropout'],
        )

        # Save state with config
        all_neural_states[cfg['name']] = {
            'state_dict': {k: v.clone() for k, v in state.items()},
            'hidden_dim': cfg['hidden_dim'],
            'spot_repr_dim': cfg['spot_repr_dim'],
            'dropout': cfg['dropout'],
        }

        model.branch_classifier.eval()
        with torch.no_grad():
            delta = model.branch_classifier.get_delta(cls_features.to(device))
            alpha_np = (0.5 + delta).cpu().numpy()

        pred = alpha_np[:, None] * ret_np + (1 - alpha_np[:, None]) * reg_np
        mse, mae, pcc = _eval_predictions(pred, gt_np)

        alpha_mean = float(alpha_np.mean())
        alpha_std = float(alpha_np.std())
        n_nonzero = int((np.abs(alpha_np - 0.5) > 0.01).sum())
        pct_nonzero = 100 * n_nonzero / len(alpha_np)

        results.append({
            'method': cfg['name'], 'type': 'neural', 'mse': mse, 'mae': mae, 'pcc': pcc,
            'alpha_info': f'mean={alpha_mean:.3f} std={alpha_std:.3f} nz={pct_nonzero:.0f}%',
        })
        logging.info(f"    {cfg['name']:<30s}  MSE={mse:.6f}  MAE={mae:.6f}  PCC={pcc:.4f}  "
                     f"alpha={alpha_mean:.3f}+/-{alpha_std:.3f}  nz={pct_nonzero:.0f}%")

        if pcc > best_neural_pcc:
            best_neural_pcc = pcc
            best_neural_method = cfg['name']
            best_neural_state = all_neural_states[cfg['name']]

    # === Per-gene alpha ===
    logging.info(f"\n  --- Per-gene alpha variants ---")

    pg_lambdas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    all_pg_alphas = {}  # Save ALL per-gene alphas
    best_pg_pcc = -float('inf')
    best_pg_method = None

    for pg_lam in pg_lambdas:
        method = f'per_gene_lam{pg_lam}'
        alpha_g = per_gene_alpha(reg_np, ret_np, gt_np, reg_lambda=pg_lam)
        all_pg_alphas[method] = alpha_g.copy()

        pred = alpha_g[None, :] * ret_np + (1 - alpha_g[None, :]) * reg_np
        mse, mae, pcc = _eval_predictions(pred, gt_np)

        n_dev = int((np.abs(alpha_g - 0.5) > 0.05).sum())
        results.append({
            'method': method, 'type': 'per_gene', 'mse': mse, 'mae': mae, 'pcc': pcc,
            'alpha_info': f'mean={alpha_g.mean():.3f} std={alpha_g.std():.3f} dev={n_dev}/{len(alpha_g)}',
        })
        logging.info(f"    {method:<30s}  MSE={mse:.6f}  MAE={mae:.6f}  PCC={pcc:.4f}  "
                     f"alpha_mean={alpha_g.mean():.3f}  dev={n_dev}/{len(alpha_g)}")

        if pcc > best_pg_pcc:
            best_pg_pcc = pcc
            best_pg_method = method

    # === Summary ===
    df = pd.DataFrame(results)
    avg_row = df[df['method'] == 'simple_avg'].iloc[0]
    avg_mse, avg_mae, avg_pcc = avg_row['mse'], avg_row['mae'], avg_row['pcc']

    df['d_mse'] = df['mse'] - avg_mse
    df['d_mae'] = df['mae'] - avg_mae
    df['d_pcc'] = df['pcc'] - avg_pcc

    logging.info(f"\n  {'='*110}")
    logging.info(f"  Stage 3 -- cls_val Comparison ({len(df)} variants, sorted by PCC)")
    logging.info(f"  {'='*110}")
    logging.info(f"  {'Method':<30s}  {'Type':<9s}  {'MSE':>9s}  {'MAE':>9s}  {'PCC':>7s}  "
                 f"{'dMSE':>9s}  {'dMAE':>9s}  {'dPCC':>7s}  {'Alpha Info'}")

    for _, r in df.sort_values('pcc', ascending=False).iterrows():
        marker = ''
        if r['method'] == best_neural_method:
            marker = ' ***BEST-NEURAL'
        elif r['method'] == best_pg_method:
            marker = ' ***BEST-PERGENE'
        elif r['method'] == 'simple_avg':
            marker = ' (baseline)'
        logging.info(
            f"  {r['method']:<30s}  {r['type']:<9s}  {r['mse']:>9.6f}  {r['mae']:>9.6f}  {r['pcc']:>7.4f}  "
            f"{r['d_mse']:>+9.6f}  {r['d_mae']:>+9.6f}  {r['d_pcc']:>+7.4f}  "
            f"{r['alpha_info']}{marker}"
        )

    logging.info(f"\n  Best neural:   {best_neural_method} (PCC={best_neural_pcc:.4f}, d={best_neural_pcc - avg_pcc:+.4f})")
    logging.info(f"  Best per-gene: {best_pg_method} (PCC={best_pg_pcc:.4f}, d={best_pg_pcc - avg_pcc:+.4f})")

    df.to_csv(os.path.join(save_path, 'stage3_cls_val.csv'), index=False)

    # Save all states to disk
    cls_dir = os.path.join(save_path, 'all_classifiers')
    os.makedirs(cls_dir, exist_ok=True)
    torch.save(all_neural_states, os.path.join(cls_dir, 'all_neural_states.pt'))
    for pg_name, pg_alpha in all_pg_alphas.items():
        np.save(os.path.join(cls_dir, f'{pg_name}.npy'), pg_alpha)

    # Restore best neural classifier into model
    from models.heads import BranchClassifier
    input_dim = model.branch_classifier.spot_encoder[0].in_features
    best_cfg = best_neural_state
    model.branch_classifier = BranchClassifier(
        input_dim=input_dim,
        hidden_dim=best_cfg['hidden_dim'],
        spot_repr_dim=best_cfg['spot_repr_dim'],
        dropout=best_cfg['dropout'],
    ).to(device)
    model.branch_classifier.load_state_dict(best_cfg['state_dict'])
    model.branch_classifier.eval()

    return best_neural_method, all_neural_states, all_pg_alphas


@torch.no_grad()
def soft_mixing_predict(model, features, device):
    """Spot-level soft mixing using trained network"""
    model.branch_classifier.eval()
    features = features.to(device)
    delta = model.branch_classifier.get_delta(features)
    alpha = 0.5 + delta
    return alpha.cpu(), delta.cpu()


# ---------------------------------------------------------------------------
# Evaluate ALL variants on val (post-training)
# ---------------------------------------------------------------------------

def evaluate_all_variants_on_val(
    model, val_loaders_by_slide, device,
    gene_embeds_db, all_target_genes, all_cell_comp,
    all_neural_states, all_pg_alphas,
    save_path, fold_idx,
    count_threshold=0.5, comp_sim_threshold=0.15, cell_sim_weight=0.3,
):
    """Evaluate ALL classifier variants on val slides"""
    from models.heads import BranchClassifier

    logging.info(f"\n{'='*70}")
    logging.info(f"Evaluating ALL variants on VAL (fold {fold_idx})")
    logging.info(f"{'='*70}")

    # Step 1: Get pred_reg, pred_ret, features, gt for all val slides
    slide_data = {}
    for slide_name, val_loader in tqdm(
        val_loaders_by_slide.items(), desc=f"Val inference fold{fold_idx}", ncols=120
    ):
        eval_result = evaluate_slide(
            model, val_loader, device,
            gene_embeds_db, all_target_genes, all_cell_comp,
            count_threshold=count_threshold,
            comp_sim_threshold=comp_sim_threshold,
            cell_sim_weight=cell_sim_weight,
        )
        slide_data[slide_name] = eval_result

    # Step 2: Define all methods to evaluate
    fixed_alphas = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    input_dim = model.branch_classifier.spot_encoder[0].in_features

    all_results = []

    for slide_name, data in slide_data.items():
        pred_reg = data['pred_reg']
        pred_ret = data['pred_ret']
        pred_ret_nf = data['pred_ret_nofilter']
        gt = data['targets']
        features = data['features']

        slide_row_base = {'slide': slide_name, 'fold': fold_idx}

        # --- Baselines ---
        m_reg = compute_metrics(pred_reg, gt)
        m_ret = compute_metrics(pred_ret, gt)
        m_ret_nf = compute_metrics(pred_ret_nf, gt)

        all_results.append({**slide_row_base, 'method': 'regression_only', 'type': 'baseline',
                            'pcc': m_reg['pcc'], 'mse': m_reg['mse'], 'mae': m_reg['mae']})
        all_results.append({**slide_row_base, 'method': 'retrieval_filtered', 'type': 'baseline',
                            'pcc': m_ret['pcc'], 'mse': m_ret['mse'], 'mae': m_ret['mae']})
        all_results.append({**slide_row_base, 'method': 'retrieval_nofilter', 'type': 'baseline',
                            'pcc': m_ret_nf['pcc'], 'mse': m_ret_nf['mse'], 'mae': m_ret_nf['mae']})

        # --- Fixed alphas ---
        for alpha_val in fixed_alphas:
            pred = alpha_val * pred_ret + (1 - alpha_val) * pred_reg
            m = compute_metrics(pred, gt)
            name = 'simple_avg' if alpha_val == 0.5 else f'fixed_alpha_{alpha_val}'
            all_results.append({**slide_row_base, 'method': name, 'type': 'fixed',
                                'pcc': m['pcc'], 'mse': m['mse'], 'mae': m['mae']})

        # --- Oracle ---
        reg_mse_spot = ((pred_reg - gt) ** 2).mean(axis=1)
        ret_mse_spot = ((pred_ret - gt) ** 2).mean(axis=1)
        oracle_pred = np.where((ret_mse_spot < reg_mse_spot)[:, None], pred_ret, pred_reg)
        m_oracle = compute_metrics(oracle_pred, gt)
        all_results.append({**slide_row_base, 'method': 'oracle_spot', 'type': 'oracle',
                            'pcc': m_oracle['pcc'], 'mse': m_oracle['mse'], 'mae': m_oracle['mae']})

        # --- Neural classifiers ---
        cls_pred_dir = os.path.join(save_path, 'test_pred', 'classifiers')
        os.makedirs(cls_pred_dir, exist_ok=True)
        for cls_name, cls_info in all_neural_states.items():
            cls = BranchClassifier(
                input_dim=input_dim,
                hidden_dim=cls_info['hidden_dim'],
                spot_repr_dim=cls_info['spot_repr_dim'],
                dropout=cls_info['dropout'],
            ).to(device)
            cls.load_state_dict(cls_info['state_dict'])
            cls.eval()

            with torch.no_grad():
                delta = cls.get_delta(features.to(device))
                alpha_np = (0.5 + delta).cpu().numpy()

            pred = alpha_np[:, None] * pred_ret + (1 - alpha_np[:, None]) * pred_reg
            m = compute_metrics(pred, gt)
            all_results.append({**slide_row_base, 'method': cls_name, 'type': 'neural',
                                'pcc': m['pcc'], 'mse': m['mse'], 'mae': m['mae'],
                                'alpha_mean': float(alpha_np.mean()), 'alpha_std': float(alpha_np.std())})

            # Save per-classifier predictions
            np.save(os.path.join(cls_pred_dir, f'{slide_name}_{cls_name}_pred.npy'), pred)
            np.save(os.path.join(cls_pred_dir, f'{slide_name}_{cls_name}_alpha.npy'), alpha_np)

        # --- Per-gene alphas ---
        for pg_name, pg_alpha in all_pg_alphas.items():
            pred = pg_alpha[None, :] * pred_ret + (1 - pg_alpha[None, :]) * pred_reg
            m = compute_metrics(pred, gt)
            all_results.append({**slide_row_base, 'method': pg_name, 'type': 'per_gene',
                                'pcc': m['pcc'], 'mse': m['mse'], 'mae': m['mae']})
            np.save(os.path.join(cls_pred_dir, f'{slide_name}_{pg_name}_pred.npy'), pred)

        # --- Save predictions for this slide ---
        val_pred_dir = os.path.join(save_path, 'test_pred')
        os.makedirs(val_pred_dir, exist_ok=True)
        np.save(os.path.join(val_pred_dir, f'{slide_name}_pred_reg.npy'), pred_reg)
        np.save(os.path.join(val_pred_dir, f'{slide_name}_pred_ret.npy'), pred_ret)
        np.save(os.path.join(val_pred_dir, f'{slide_name}_pred_ret_nofilter.npy'), pred_ret_nf)
        np.save(os.path.join(val_pred_dir, f'{slide_name}_pred_avg.npy'),
                0.5 * pred_reg + 0.5 * pred_ret)
        np.save(os.path.join(val_pred_dir, f'{slide_name}_gt.npy'), gt)
        torch.save(features, os.path.join(val_pred_dir, f'{slide_name}_features.pt'))
        # Best classifier fused prediction
        with torch.no_grad():
            alpha_best, _ = soft_mixing_predict(model, features, device)
            alpha_best_np = alpha_best.numpy()
        cls_pred = alpha_best_np[:, None] * pred_ret + (1 - alpha_best_np[:, None]) * pred_reg
        np.save(os.path.join(val_pred_dir, f'{slide_name}_pred_cls.npy'), cls_pred)
        np.save(os.path.join(val_pred_dir, f'{slide_name}_alpha.npy'), alpha_best_np)

    df = pd.DataFrame(all_results)

    # --- Summary: average across slides ---
    logging.info(f"\n  {'='*90}")
    logging.info(f"  VAL Results -- All Variants (fold {fold_idx}, {len(slide_data)} slides)")
    logging.info(f"  {'='*90}")

    summary = df.groupby('method')[['pcc', 'mse', 'mae']].mean().reset_index()
    avg_pcc = summary.loc[summary['method'] == 'simple_avg', 'pcc'].values[0]
    summary['d_pcc'] = summary['pcc'] - avg_pcc
    summary = summary.sort_values('pcc', ascending=False)

    logging.info(f"  {'Method':<30s}  {'PCC':>8s}  {'MSE':>9s}  {'MAE':>9s}  {'dPCC':>8s}")
    logging.info(f"  {'-'*30}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*8}")
    for _, r in summary.iterrows():
        logging.info(f"  {r['method']:<30s}  {r['pcc']:>8.4f}  {r['mse']:>9.6f}  {r['mae']:>9.6f}  {r['d_pcc']:>+8.4f}")

    # Save
    val_eval_dir = os.path.join(save_path, 'val_all_variants')
    os.makedirs(val_eval_dir, exist_ok=True)
    df.to_csv(os.path.join(val_eval_dir, 'per_slide.csv'), index=False)
    summary.to_csv(os.path.join(val_eval_dir, 'summary.csv'), index=False)
    logging.info(f"\n  Saved to {val_eval_dir}/")

    return df


# ---------------------------------------------------------------------------
# Merge utilities
# ---------------------------------------------------------------------------

def merge_epoch_metrics_to_excel(root_dir, save_dir):
    """Merge slidewise metrics from all folds into per-epoch Excel files"""
    all_folds = sorted(glob(os.path.join(root_dir, "fold*")))
    if not all_folds:
        return

    epoch_dict = {}
    for fold_dir in all_folds:
        fold_idx = int(os.path.basename(fold_dir).replace("fold", ""))
        metrics_dir = os.path.join(fold_dir, "slidewise_metrics")
        if not os.path.isdir(metrics_dir):
            continue
        for csv_file in glob(os.path.join(metrics_dir, "epoch_*.csv")):
            epoch_name = os.path.splitext(os.path.basename(csv_file))[0]
            if epoch_name not in epoch_dict:
                epoch_dict[epoch_name] = []
            df = pd.read_csv(csv_file)
            df["fold"] = fold_idx
            epoch_dict[epoch_name].append(df)

    if not epoch_dict:
        return

    os.makedirs(save_dir, exist_ok=True)
    for epoch_name, df_list in epoch_dict.items():
        df_epoch = pd.concat(df_list, ignore_index=True)
        epoch_num = epoch_name.split('_')[1]
        xlsx_path = os.path.join(save_dir, f"epoch_{epoch_num}.xlsx")
        df_epoch.to_excel(xlsx_path, index=False)
        logging.info(f"[+] Merged metrics saved to {xlsx_path}")


# ---------------------------------------------------------------------------
# Main fold runner
# ---------------------------------------------------------------------------

def run_fold(args, fold_idx):
    """Run 3-stage training for one fold"""
    logging.info(f"\n{'='*70}")
    logging.info(f"Fold {fold_idx} | Dataset: {args.dataset} | Mask: {args.mask_ratio}")
    logging.info(f"Stage 1 (BLEEP): {args.stage1_epochs} ep | Stage 2 (Reg): {args.stage2_epochs} ep")
    logging.info(f"Temperature: {args.temperature} | Top-K: {args.top_k}")
    logging.info(f"{'='*70}\n")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Get data splits
    label_root = os.path.join(DATA_ROOT, args.dataset)
    folds = get_folds(args.dataset)
    split = folds[fold_idx]
    train_files = [os.path.join(label_root, f) for f in split["train"]]
    val_files = [os.path.join(label_root, f) for f in split["val"]]
    cls_val_files = [os.path.join(label_root, f) for f in split.get("cls_val", [])]

    train_loader, val_loaders_by_slide, train_samples, cls_val_loaders_by_slide, cls_val_samples = \
        create_dataloaders(
            train_npy_files=train_files,
            val_npy_files=val_files,
            dataset_name=args.dataset,
            mask_ratio=args.mask_ratio,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cls_val_npy_files=cls_val_files if cls_val_files else None,
            hvg_k=args.hvg_k,
        )

    # Build retrieval database
    logging.info("Building retrieval database...")
    target_indices = train_loader.dataset.target_indices
    all_target_genes = torch.stack([
        torch.tensor(s['expression_norm'][target_indices], dtype=torch.float32)
        for s in train_samples
    ])
    n_cell_types = DATASET_CONFIG[args.dataset]['n_cell_types']
    cell_counts_key = 'cell_type_counts' if args.hvg_k == 2000 else f'cell_type_counts_hvg{args.hvg_k}'
    all_cell_comp = torch.stack([
        torch.tensor(
            s.get(cell_counts_key, s.get('cell_type_counts', np.zeros(n_cell_types))),
            dtype=torch.float32
        )
        for s in train_samples
    ])
    logging.info(f"Database: {all_target_genes.shape[0]} samples, {all_target_genes.shape[1]} genes")

    _, output_dim = get_model_dims(args.dataset, args.mask_ratio)

    # Save paths
    if hasattr(args, 'weight_dir') and args.weight_dir:
        base = f"{args.weight_dir}/{args.dataset}/mask_{int(args.mask_ratio) if args.mask_ratio >= 1 else args.mask_ratio}"
    else:
        base = f"./weights/{args.dataset}/mask_{int(args.mask_ratio) if args.mask_ratio >= 1 else args.mask_ratio}"
    save_path = os.path.join(base, f"fold{fold_idx}")
    save_pred_path = os.path.join(save_path, "pred")
    save_gt_path = os.path.join(save_path, "gt")
    save_metrics_path = os.path.join(save_path, "slidewise_metrics")

    for p in [save_path, save_pred_path, save_gt_path, save_metrics_path]:
        os.makedirs(p, exist_ok=True)

    writer = SummaryWriter(os.path.join(save_path, "log"))

    # Create model
    model = DualBackboneModel(
        n_target_genes=output_dim,
        projection_dim=args.projection_dim,
        temperature=args.temperature,
        top_k=args.top_k,
        pretrained=True,
    )
    model = model.to(device)

    if fold_idx == 0:
        model.print_parameter_summary()

    start_time = time.time()
    total_epochs = args.stage1_epochs + args.stage2_epochs

    # ===================================================================
    # Stage 1: BLEEP (DenseNet-121 + GeneEncoder)
    # ===================================================================
    logging.info(f"\n{'='*70}")
    logging.info("STAGE 1: Training DenseNet-121 + GeneEncoder (BLEEP contrastive)")
    logging.info(f"  Temperature: {args.temperature}, Projection: {args.projection_dim}")
    logging.info(f"{'='*70}")

    optimizer_bleep = AdamW(
        model.get_optimizer_params_bleep(),
        lr=args.lr_bleep, weight_decay=args.weight_decay
    )
    scheduler_bleep = CosineAnnealingLR(
        optimizer_bleep, T_max=args.stage1_epochs, eta_min=1e-6
    )

    best_ret_pcc = -float('inf')
    patience_counter = 0
    best_stage1_path = os.path.join(save_path, 'best_stage1.pt')

    for epoch in range(args.stage1_epochs):
        global_epoch = epoch + 1
        logging.info(f"\n[Stage 1] Epoch {global_epoch}/{args.stage1_epochs}")

        train_metrics = train_epoch_stage1(model, train_loader, optimizer_bleep, device)
        scheduler_bleep.step()

        writer.add_scalar("stage1/loss_contrastive", train_metrics['loss_con'], global_epoch)
        logging.info(f"  Contrastive loss: {train_metrics['loss_con']:.4f}")

        # Evaluate every 10 epochs or last epoch
        if global_epoch % 10 == 0 or epoch == args.stage1_epochs - 1:
            gene_embeds_db = build_gene_embed_database(
                model, all_target_genes, batch_size=256, device=device
            )

            results = []
            for slide_name, val_loader in tqdm(
                val_loaders_by_slide.items(),
                desc=f"Eval S1 Fold {fold_idx} | Epoch {global_epoch}",
                ncols=120
            ):
                eval_result = evaluate_retrieval(
                    model, val_loader, device,
                    gene_embeds_db, all_target_genes,
                    db_cell_comp=all_cell_comp,
                    count_threshold=args.count_threshold,
                    comp_sim_threshold=args.comp_sim_threshold,
                    cell_sim_weight=args.cell_sim_weight,
                )

                np.save(
                    os.path.join(save_gt_path, f"{slide_name}_gt.npy"),
                    eval_result['targets']
                )

                ret_nf = eval_result['ret_nofilter']
                ret_f = eval_result.get('ret_filtered', {'mse': float('nan'), 'mae': float('nan'), 'pcc': float('nan')})

                logging.info(
                    f"  [{slide_name}] "
                    f"Ret(no filter) PCC: {ret_nf['pcc']:.4f} | "
                    f"Ret(filtered) PCC: {ret_f['pcc']:.4f}"
                )

                results.append({
                    "slide": slide_name,
                    "epoch": global_epoch,
                    "stage": 1,
                    "ret_nf_mse": ret_nf['mse'],
                    "ret_nf_pcc": ret_nf['pcc'],
                    "ret_f_mse": ret_f['mse'],
                    "ret_f_pcc": ret_f['pcc'],
                })

            df = pd.DataFrame(results)
            df.to_csv(
                os.path.join(save_metrics_path, f"epoch_{global_epoch}.csv"),
                index=False
            )

            avg_ret_nf_pcc = df['ret_nf_pcc'].mean()
            avg_ret_f_pcc = df['ret_f_pcc'].mean()
            writer.add_scalar("val/ret_nofilter_pcc", avg_ret_nf_pcc, global_epoch)
            writer.add_scalar("val/ret_filtered_pcc", avg_ret_f_pcc, global_epoch)

            logging.info(f"  Stage 1 Epoch {global_epoch}: "
                         f"Ret(no filter) PCC={avg_ret_nf_pcc:.4f} | "
                         f"Ret(filtered) PCC={avg_ret_f_pcc:.4f}")

            # Use filtered PCC for best model selection
            current_pcc = avg_ret_f_pcc if not np.isnan(avg_ret_f_pcc) else avg_ret_nf_pcc
            if current_pcc > best_ret_pcc:
                best_ret_pcc = current_pcc
                # Save BLEEP components only
                torch.save({
                    'bleep_encoder': model.bleep_encoder.state_dict(),
                    'bleep_projection': model.bleep_projection.state_dict(),
                    'gene_encoder': model.gene_encoder.state_dict(),
                }, best_stage1_path)
                logging.info(f"    *** Best Stage 1 (Ret PCC: {best_ret_pcc:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1

            if args.patience > 0 and patience_counter >= args.patience:
                logging.info(f"  Early stopping at epoch {global_epoch} (patience={args.patience})")
                break

    # Load best Stage 1 and freeze BLEEP
    logging.info(f"\nLoading best Stage 1 BLEEP (Ret PCC: {best_ret_pcc:.4f})")
    ckpt = torch.load(best_stage1_path, map_location=device)
    model.bleep_encoder.load_state_dict(ckpt['bleep_encoder'])
    model.bleep_projection.load_state_dict(ckpt['bleep_projection'])
    model.gene_encoder.load_state_dict(ckpt['gene_encoder'])
    model.freeze_bleep()
    os.remove(best_stage1_path)
    logging.info("BLEEP branch frozen.")

    # ===================================================================
    # Stage 2: Regression (ViT-B + RegressionHead)
    # ===================================================================
    logging.info(f"\n{'='*70}")
    logging.info("STAGE 2: Training ViT-B (CONCH) + RegressionHead (MSE)")
    if args.lambda_ret > 0:
        logging.info(f"  + Retrieval guidance: MSE(pred_reg, pred_ret), lambda={args.lambda_ret}")
    logging.info(f"{'='*70}")

    optimizer_reg = AdamW(
        model.get_optimizer_params_reg(),
        lr=args.lr_reg, weight_decay=args.weight_decay
    )
    scheduler_reg = CosineAnnealingLR(
        optimizer_reg, T_max=args.stage2_epochs, eta_min=1e-6
    )

    best_reg_pcc = -float('inf')
    patience_counter = 0
    best_stage2_path = os.path.join(save_path, 'best_stage2.pt')

    # Build gene embed db for full evaluation
    gene_embeds_db = build_gene_embed_database(
        model, all_target_genes, batch_size=256, device=device
    )

    for epoch in range(args.stage2_epochs):
        global_epoch = args.stage1_epochs + epoch + 1

        # Cosine decay over first 30 epochs, then 0
        import math
        if epoch < 30:
            current_lambda_ret = args.lambda_ret * 0.5 * (1.0 + math.cos(math.pi * epoch / 30))
        else:
            current_lambda_ret = 0.0

        logging.info(f"\n[Stage 2] Epoch {global_epoch}/{total_epochs}  lambda_ret={current_lambda_ret:.4f}")

        train_metrics = train_epoch_stage2(
            model, train_loader, optimizer_reg, device,
            gene_embeds_db=gene_embeds_db if args.lambda_ret > 0 else None,
            target_genes_db=all_target_genes if args.lambda_ret > 0 else None,
            db_cell_comp=all_cell_comp if args.lambda_ret > 0 else None,
            lambda_ret=current_lambda_ret,
            count_threshold=args.count_threshold,
            comp_sim_threshold=args.comp_sim_threshold,
            cell_sim_weight=args.cell_sim_weight,
        )
        scheduler_reg.step()

        writer.add_scalar("stage2/loss_mse_gt", train_metrics['loss_reg'], global_epoch)
        if 'loss_ret' in train_metrics:
            writer.add_scalar("stage2/loss_mse_ret", train_metrics['loss_ret'], global_epoch)
            writer.add_scalar("stage2/lambda_ret", current_lambda_ret, global_epoch)
            logging.info(f"  MSE_gt: {train_metrics['loss_reg']:.4f}  MSE_ret: {train_metrics['loss_ret']:.4f}")
        else:
            logging.info(f"  MSE: {train_metrics['loss_reg']:.4f}")

        # Evaluate every 10 epochs or last epoch
        if global_epoch % 10 == 0 or epoch == args.stage2_epochs - 1:
            results = []
            for slide_name, val_loader in tqdm(
                val_loaders_by_slide.items(),
                desc=f"Eval S2 Fold {fold_idx} | Epoch {global_epoch}",
                ncols=120
            ):
                eval_result = evaluate_slide(
                    model, val_loader, device,
                    gene_embeds_db, all_target_genes, all_cell_comp,
                    count_threshold=args.count_threshold,
                    comp_sim_threshold=args.comp_sim_threshold,
                    cell_sim_weight=args.cell_sim_weight,
                )

                # Save predictions
                epdir = os.path.join(save_pred_path, f"epoch_{global_epoch}")
                os.makedirs(epdir, exist_ok=True)
                np.save(os.path.join(epdir, f"{slide_name}_pred_reg.npy"), eval_result['pred_reg'])
                np.save(os.path.join(epdir, f"{slide_name}_pred_ret.npy"), eval_result['pred_ret'])
                np.save(os.path.join(epdir, f"{slide_name}_pred_ret_nofilter.npy"), eval_result['pred_ret_nofilter'])
                np.save(os.path.join(save_gt_path, f"{slide_name}_gt.npy"), eval_result['targets'])
                torch.save(eval_result['features'], os.path.join(epdir, f"{slide_name}_features.pt"))

                logging.info(
                    f"  [{slide_name}] "
                    f"Reg PCC: {eval_result['regression']['pcc']:.4f} | "
                    f"Ret PCC: {eval_result['retrieval']['pcc']:.4f} | "
                    f"Ret(nf) PCC: {eval_result['retrieval_nofilter']['pcc']:.4f}"
                )

                results.append({
                    "slide": slide_name,
                    "epoch": global_epoch,
                    "stage": 2,
                    "reg_mse": eval_result['regression']['mse'],
                    "reg_pcc": eval_result['regression']['pcc'],
                    "ret_mse": eval_result['retrieval']['mse'],
                    "ret_pcc": eval_result['retrieval']['pcc'],
                    "ret_nf_mse": eval_result['retrieval_nofilter']['mse'],
                    "ret_nf_pcc": eval_result['retrieval_nofilter']['pcc'],
                })

            df = pd.DataFrame(results)
            df.to_csv(
                os.path.join(save_metrics_path, f"epoch_{global_epoch}.csv"),
                index=False
            )

            avg_reg_pcc = df['reg_pcc'].mean()
            avg_ret_pcc = df['ret_pcc'].mean()
            avg_ret_nf_pcc = df['ret_nf_pcc'].mean()
            writer.add_scalar("val/reg_pcc", avg_reg_pcc, global_epoch)
            writer.add_scalar("val/ret_filtered_pcc", avg_ret_pcc, global_epoch)
            writer.add_scalar("val/ret_nofilter_pcc", avg_ret_nf_pcc, global_epoch)

            logging.info(f"  Stage 2 Epoch {global_epoch}: "
                         f"Reg PCC={avg_reg_pcc:.4f} | "
                         f"Ret PCC={avg_ret_pcc:.4f} | "
                         f"Ret(nf) PCC={avg_ret_nf_pcc:.4f}")

            if avg_reg_pcc > best_reg_pcc:
                best_reg_pcc = avg_reg_pcc
                torch.save({
                    'reg_encoder': model.reg_encoder.state_dict(),
                    'regression_head': model.regression_head.state_dict(),
                }, best_stage2_path)
                logging.info(f"    *** Best Stage 2 (Reg PCC: {best_reg_pcc:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1

            if args.patience > 0 and patience_counter >= args.patience:
                logging.info(f"  Early stopping at epoch {global_epoch} (patience={args.patience})")
                break

    # Load best Stage 2 and freeze regression
    logging.info(f"\nLoading best Stage 2 (Reg PCC: {best_reg_pcc:.4f})")
    ckpt = torch.load(best_stage2_path, map_location=device)
    model.reg_encoder.load_state_dict(ckpt['reg_encoder'])
    model.regression_head.load_state_dict(ckpt['regression_head'])
    model.freeze_regression()
    os.remove(best_stage2_path)
    logging.info("Regression branch frozen.")

    # ===================================================================
    # Stage 3: Soft Mixing
    # ===================================================================
    logging.info(f"\n{'='*70}")
    logging.info("STAGE 3: Training soft mixing -- all variants")
    logging.info(f"{'='*70}")

    if cls_val_loaders_by_slide is None or len(cls_val_loaders_by_slide) == 0:
        logging.warning("  No cls_val slides available, skipping Stage 3")
    else:
        gene_embeds_db = build_gene_embed_database(
            model, all_target_genes, batch_size=256, device=device
        )

        cls_features, cls_pred_reg, cls_pred_ret, cls_gt = \
            generate_soft_mixing_data(
                model=model,
                cls_val_loaders_by_slide=cls_val_loaders_by_slide,
                device=device,
                gene_embeds_db=gene_embeds_db,
                all_target_genes=all_target_genes,
                all_cell_comp=all_cell_comp,
                count_threshold=args.count_threshold,
                comp_sim_threshold=args.comp_sim_threshold,
                cell_sim_weight=args.cell_sim_weight,
            )

        best_method, all_neural_states, all_pg_alphas = run_stage3_all_variants(
            model=model,
            cls_features=cls_features,
            cls_pred_reg=cls_pred_reg,
            cls_pred_ret=cls_pred_ret,
            cls_gt=cls_gt,
            device=device,
            save_path=save_path,
        )
        logging.info(f"  Selected classifier: {best_method}")

    # Save final model
    torch.save({
        'epoch': total_epochs,
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'best_reg_pcc': best_reg_pcc,
        'best_ret_pcc': best_ret_pcc,
    }, os.path.join(save_path, 'final_model.pt'))

    # Evaluate ALL variants on val
    if cls_val_loaders_by_slide is not None and len(cls_val_loaders_by_slide) > 0:
        evaluate_all_variants_on_val(
            model=model,
            val_loaders_by_slide=val_loaders_by_slide,
            device=device,
            gene_embeds_db=gene_embeds_db,
            all_target_genes=all_target_genes,
            all_cell_comp=all_cell_comp,
            all_neural_states=all_neural_states,
            all_pg_alphas=all_pg_alphas,
            save_path=save_path,
            fold_idx=fold_idx,
            count_threshold=args.count_threshold,
            comp_sim_threshold=args.comp_sim_threshold,
            cell_sim_weight=args.cell_sim_weight,
        )

    elapsed = (time.time() - start_time) / 60
    logging.info(f"\nFold {fold_idx} completed in {elapsed:.1f} minutes")
    logging.info(f"  Best BLEEP Ret PCC (Stage 1): {best_ret_pcc:.4f}")
    logging.info(f"  Best Reg PCC (Stage 2): {best_reg_pcc:.4f}")
    writer.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dual-Backbone Gene Expression Prediction")

    # Data
    parser.add_argument("--dataset", choices=["HER2", "BC", "Kidney"], required=True)
    parser.add_argument("--mask_ratio", type=float, required=True)
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override DATA_ROOT (default: config.py DATA_ROOT)")

    # Model
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--top_k", type=int, default=150)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_bleep", type=float, default=1e-3,
                        help="Learning rate for BLEEP (DenseNet + GeneEncoder)")
    parser.add_argument("--lr_reg", type=float, default=1e-4,
                        help="Learning rate for regression (ViT-B + RegHead)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--stage1_epochs", type=int, default=150,
                        help="Max epochs for Stage 1 (BLEEP)")
    parser.add_argument("--stage2_epochs", type=int, default=50,
                        help="Max epochs for Stage 2 (Regression)")
    parser.add_argument("--lambda_ret", type=float, default=0.1,
                        help="Weight for MSE(pred_reg, pred_ret) guidance loss in Stage 2 (0=disabled)")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience (0=disabled)")

    # Cell composition filtering
    parser.add_argument("--count_threshold", type=float, default=0.5)
    parser.add_argument("--comp_sim_threshold", type=float, default=0.15)
    parser.add_argument("--cell_sim_weight", type=float, default=0.3)
    parser.add_argument("--hvg_k", type=int, default=2000,
                        help="Number of HVGs used in cell deconv (default: 2000). "
                             "Controls which cell_type_counts key to load from npy.")

    # System
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--weight_dir", type=str, default=None)
    parser.add_argument("--fold", type=int, default=None,
                        help="Run single fold (0-3). If not specified, run all 4.")

    args = parser.parse_args()

    # Override DATA_ROOT if specified
    global DATA_ROOT
    if args.data_root:
        DATA_ROOT = args.data_root

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.info(f"\n{'='*70}")
    logging.info("Dual-Backbone Gene Expression Prediction")
    logging.info(f"{'='*70}")
    logging.info(f"Dataset: {args.dataset} | Mask: {args.mask_ratio}")
    logging.info(f"BLEEP: DenseNet-121, temp={args.temperature}, top_k={args.top_k}")
    logging.info(f"Regression: ViT-B (CONCH)")
    logging.info(f"Stage 1 (BLEEP): {args.stage1_epochs} ep, lr={args.lr_bleep}")
    logging.info(f"Stage 2 (Reg):   {args.stage2_epochs} ep, lr={args.lr_reg}, lambda_ret={args.lambda_ret}")
    logging.info(f"Cell deconv HVG: {args.hvg_k}")
    if args.data_root:
        logging.info(f"Data root: {args.data_root}")
    logging.info(f"{'='*70}")

    if args.weight_dir:
        root_dir = f"{args.weight_dir}/{args.dataset}/mask_{int(args.mask_ratio) if args.mask_ratio >= 1 else args.mask_ratio}"
    else:
        root_dir = f"./weights/{args.dataset}/mask_{int(args.mask_ratio) if args.mask_ratio >= 1 else args.mask_ratio}"
    total_epochs = args.stage1_epochs + args.stage2_epochs

    # Determine which folds to run
    if args.fold is not None:
        fold_list = [args.fold]
        logging.info(f"Running single fold: {args.fold}")
    else:
        fold_list = list(range(4))

    for fold_idx in fold_list:
        run_fold(args, fold_idx)

        logging.info(f"\n{'='*70}")
        logging.info(f"FOLD {fold_idx} COMPLETED")
        logging.info(f"{'='*70}")

        merge_epoch_metrics_to_excel(
            root_dir=root_dir,
            save_dir=os.path.join(root_dir, "merged_metrics")
        )

    logging.info("\n" + "="*70)
    logging.info("All done!")
    logging.info("="*70)


if __name__ == "__main__":
    main()
