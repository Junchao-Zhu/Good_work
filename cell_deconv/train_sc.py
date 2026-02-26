# -*- coding: utf-8 -*-
"""
Cell2location SC regression model training.

Train cell type signature models from single-cell RNA-seq data.
Supports breast and kidney tissue types.

Usage:
    python train_sc.py --tissue breast
    python train_sc.py --tissue kidney
"""

import scanpy as sc
import numpy as np
import pandas as pd
import os
import argparse
import time
from datetime import timedelta
from scipy.sparse import issparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Timer utilities
# ============================================================

START_TIME = time.time()
STEP_TIMES = {}


def print_step(step_name, step_num=None, total_steps=None):
    elapsed = time.time() - START_TIME
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    if step_num and total_steps:
        print(f"\n{'='*60}")
        print(f"[{elapsed_str}] Step {step_num}/{total_steps}: {step_name}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"[{elapsed_str}] {step_name}")
        print(f"{'='*60}")
    STEP_TIMES[step_name] = time.time()


def print_progress(msg):
    elapsed = time.time() - START_TIME
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    print(f"[{elapsed_str}] {msg}")


# ============================================================
# Tissue-specific configurations
# ============================================================

TISSUE_CONFIGS = {
    "breast": {
        "sc_data_dir": "./sc_data/breast_sc_data/",
        "sc_files": [
            "09595871-5cde-4351-88f9-b3c37b3ed466.h5ad",  # 355k cells
            "303bc6a5-0811-4ee6-97b8-399f883ce0a2.h5ad",  # 621k cells
            "fac7ca72-8814-42b4-9671-e266db7b913d.h5ad",  # 2.1M cells
        ],
        "results_folder": "./pretrained/breast/",
        "batch_size": 25000,
    },
    "kidney": {
        "sc_data_dir": "./sc_data/kidney_sc_data/",
        "sc_files": [
            "09595871-5cde-4351-88f9-b3c37b3ed466.h5ad",
            "f337b525-c8f7-4c96-8cfe-f258a9f5ca48.h5ad",
        ],
        "results_folder": "./pretrained/kidney/",
        "batch_size": 12500,
    },
}

# Common parameters
CELLTYPE_COLUMN = 'cell_type'
BATCH_COLUMN = 'donor_id'
MAX_EPOCHS = 200
SAVE_EVERY = 40


# ============================================================
# Data loading and preparation
# ============================================================

def check_raw_counts(X, name="data"):
    """Check whether data contains raw counts."""
    print(f"\n  === Raw Count Check ({name}) ===")

    if issparse(X):
        X_sample = X[:min(1000, X.shape[0]), :].toarray()
    else:
        X_sample = X[:min(1000, X.shape[0]), :]

    print(f"  dtype: {X_sample.dtype}")
    print(f"  min: {X_sample.min():.4f}")
    print(f"  max: {X_sample.max():.4f}")
    print(f"  mean: {X_sample.mean():.4f}")

    is_integer = np.allclose(X_sample, np.round(X_sample))
    print(f"  integer values: {is_integer}")

    has_negative = (X_sample < 0).any()
    print(f"  has negatives: {has_negative}")

    nonzero_vals = X_sample[X_sample > 0]
    if len(nonzero_vals) > 0:
        print(f"  nonzero count: {len(nonzero_vals)}")
        print(f"  nonzero mean: {nonzero_vals.mean():.4f}")
        print(f"  nonzero median: {np.median(nonzero_vals):.4f}")
        print(f"  nonzero max: {nonzero_vals.max():.4f}")

    is_raw_counts = is_integer and not has_negative and X_sample.max() > 10

    if is_raw_counts:
        print(f"  Conclusion: data appears to be RAW COUNTS")
    elif not is_integer:
        print(f"  Conclusion: data is normalized (non-integer)")
    elif has_negative:
        print(f"  Conclusion: data is log-transformed (has negatives)")
    elif X_sample.max() <= 10:
        print(f"  Conclusion: data may be normalized (max too small)")
    else:
        print(f"  Conclusion: uncertain, please check")

    return is_raw_counts


def load_and_prepare_sc_data(file_path, dataset_name, file_idx, total_files):
    """Load a single SC dataset and prepare for merging."""
    print(f"\n  --- Loading file {file_idx}/{total_files}: {dataset_name} ---")
    load_start = time.time()

    print_progress(f"  Reading {dataset_name}...")
    adata = sc.read_h5ad(file_path)
    load_time = time.time() - load_start
    print_progress(f"  Done! Time: {load_time:.1f}s")
    print(f"  Shape: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    if CELLTYPE_COLUMN not in adata.obs.columns:
        raise ValueError(f"Missing cell type column: {CELLTYPE_COLUMN}")
    if BATCH_COLUMN not in adata.obs.columns:
        raise ValueError(f"Missing batch column: {BATCH_COLUMN}")

    if 'feature_name' in adata.var.columns:
        adata.var['SYMBOL'] = adata.var['feature_name']
    else:
        adata.var['SYMBOL'] = adata.var_names

    is_main_raw = check_raw_counts(adata.X, "main matrix adata.X")

    has_raw_layer = adata.raw is not None
    print(f"\n  Has raw layer: {has_raw_layer}")

    if has_raw_layer:
        is_raw_layer_raw = check_raw_counts(adata.raw.X, "raw layer adata.raw.X")
        if not is_main_raw and is_raw_layer_raw:
            print("\n  Main matrix is normalized, using raw layer counts...")
            adata = adata.raw.to_adata()
    elif not is_main_raw:
        print("\n  WARNING: Data is normalized and no raw layer! Cell2location needs raw counts!")
        print("  Continuing with current data, results may be inaccurate...")

    adata.obs['dataset'] = dataset_name
    adata.obs['batch_unique'] = dataset_name + '_' + adata.obs[BATCH_COLUMN].astype(str)

    print(f"\n  Cell types: {adata.obs[CELLTYPE_COLUMN].nunique()}")
    print(f"  Batches: {adata.obs['batch_unique'].nunique()}")

    return adata


def merge_sc_datasets(adata_list):
    """Merge multiple SC datasets using common genes."""
    print("\nDataset info:")
    total_cells = 0
    for i, adata in enumerate(adata_list):
        print(f"  Dataset {i+1}: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")
        total_cells += adata.shape[0]
    print(f"  Total: {total_cells:,} cells")

    print_progress("Finding common genes...")
    common_genes = set(adata_list[0].var_names)
    for adata in adata_list[1:]:
        common_genes = common_genes.intersection(set(adata.var_names))
    common_genes = sorted(list(common_genes))

    print(f"Common genes: {len(common_genes):,}")

    print_progress("Subsetting to common genes...")
    adata_subsets = []
    for i, adata in enumerate(adata_list):
        adata_subset = adata[:, common_genes].copy()
        adata_subsets.append(adata_subset)
        print_progress(f"  Dataset {i+1}/{len(adata_list)} done")

    print_progress("Concatenating datasets...")
    adata_merged = sc.concat(adata_subsets, join='outer', label='dataset_idx')

    print(f"\nMerged shape: {adata_merged.shape[0]:,} cells x {adata_merged.shape[1]:,} genes")
    print(f"Total cell types: {adata_merged.obs[CELLTYPE_COLUMN].nunique()}")
    print(f"Total batches: {adata_merged.obs['batch_unique'].nunique()}")

    print("\nCell type distribution (top 15):")
    cell_type_counts = adata_merged.obs[CELLTYPE_COLUMN].value_counts()
    for ct, count in cell_type_counts.head(15).items():
        print(f"  {ct}: {count:,} ({count/len(adata_merged)*100:.1f}%)")

    return adata_merged


# ============================================================
# Main training pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Cell2location SC regression training')
    parser.add_argument('--tissue', type=str, required=True, choices=['breast', 'kidney'],
                        help='Tissue type (breast or kidney)')
    args = parser.parse_args()

    config = TISSUE_CONFIGS[args.tissue]
    SC_DATA_DIR = config['sc_data_dir']
    SC_FILES = config['sc_files']
    BATCH_SIZE = config['batch_size']
    results_folder = config['results_folder']

    os.makedirs(results_folder, exist_ok=True)

    # GPU setup - must be before importing cell2location
    os.environ["THEANO_FLAGS"] = 'device=cuda,floatX=float32,force_device=True'

    import torch
    torch.set_float32_matmul_precision('high')

    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"       Memory: {props.total_memory / 1024**3:.1f} GB")

    print_progress("Importing cell2location...")
    import cell2location
    from cell2location.utils.filtering import filter_genes
    from cell2location.models import RegressionModel

    from matplotlib import rcParams
    import matplotlib.pyplot as plt
    rcParams['pdf.fonttype'] = 42

    print_progress("Environment initialized!")
    print(f"\nPipeline: 1.Load data -> 2.Merge -> 3.Gene filter -> 4.Train -> 5.Export")

    # ============================================================
    # Step 1: Load SC data (parallel)
    # ============================================================

    print_step("Loading SC datasets (parallel)", 1, 5)
    print(f"Files to load: {len(SC_FILES)}")

    def load_single_file(task_args):
        file_path, dataset_name, file_idx, total_files = task_args
        return load_and_prepare_sc_data(file_path, dataset_name, file_idx, total_files)

    load_tasks = []
    for idx, sc_file in enumerate(SC_FILES, 1):
        file_path = os.path.join(SC_DATA_DIR, sc_file)
        dataset_name = sc_file.split('.')[0][:8]
        load_tasks.append((file_path, dataset_name, idx, len(SC_FILES)))

    adata_list = []
    with ThreadPoolExecutor(max_workers=len(SC_FILES)) as executor:
        futures = {executor.submit(load_single_file, task): task[1] for task in load_tasks}
        for future in as_completed(futures):
            dataset_name = futures[future]
            try:
                adata = future.result()
                adata_list.append(adata)
                print_progress(f"  {dataset_name} loaded! ({len(adata_list)}/{len(SC_FILES)})")
            except Exception as e:
                print(f"  Failed to load {dataset_name}: {e}")

    adata_list.sort(key=lambda x: x.obs['dataset'].iloc[0])

    # ============================================================
    # Step 2: Merge datasets
    # ============================================================

    print_step("Merging datasets", 2, 5)
    merge_start = time.time()
    adata_ref = merge_sc_datasets(adata_list)
    print_progress(f"Merge done! Time: {time.time() - merge_start:.1f}s")

    del adata_list
    import gc
    gc.collect()

    # ============================================================
    # Step 3: Gene filtering
    # ============================================================

    print_step("Gene filtering", 3, 5)
    print(f"Genes before filtering: {adata_ref.shape[1]:,}")
    print_progress("Filtering genes...")

    filter_start = time.time()
    selected = filter_genes(
        adata_ref,
        cell_count_cutoff=5,
        cell_percentage_cutoff2=0.03,
        nonz_mean_cutoff=1.12
    )

    adata_ref = adata_ref[:, selected].copy()
    print_progress(f"Gene filtering done! Time: {time.time() - filter_start:.1f}s")
    print(f"Genes after filtering: {adata_ref.shape[1]:,}")

    # ============================================================
    # Step 4: Train regression model
    # ============================================================

    print_step("Training regression model", 4, 5)

    print_progress("Setting up AnnData...")
    RegressionModel.setup_anndata(
        adata=adata_ref,
        batch_key='batch_unique',
        labels_key=CELLTYPE_COLUMN,
    )

    print_progress("Creating model...")
    mod = RegressionModel(adata_ref)
    mod.view_anndata_setup()

    print(f"\nTraining config:")
    print(f"  - Cells: {adata_ref.shape[0]:,}")
    print(f"  - Genes: {adata_ref.shape[1]:,}")
    print(f"  - Cell types: {adata_ref.obs[CELLTYPE_COLUMN].nunique()}")
    print(f"  - Batches: {adata_ref.obs['batch_unique'].nunique()}")
    print(f"  - Max Epochs: {MAX_EPOCHS}")
    print(f"  - Batch Size: {BATCH_SIZE:,}")

    print_progress("Starting training...")
    train_start = time.time()

    epochs_trained = 0
    for checkpoint_idx in range(1, MAX_EPOCHS // SAVE_EVERY + 1):
        epochs_to_train = SAVE_EVERY

        train_kwargs = {
            'max_epochs': epochs_to_train,
            'accelerator': 'gpu',
            'batch_size': BATCH_SIZE,
        }

        print(f"\n{'='*40}")
        print(f"Phase {checkpoint_idx}/{MAX_EPOCHS // SAVE_EVERY}: "
              f"Epoch {epochs_trained + 1} - {epochs_trained + epochs_to_train}")
        print(f"{'='*40}")

        mod.train(**train_kwargs)
        epochs_trained += epochs_to_train

        # Save checkpoint
        checkpoint_dir = f"{results_folder}/checkpoint_epoch_{epochs_trained}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        print_progress(f"Saving Epoch {epochs_trained} checkpoint...")

        adata_checkpoint = mod.export_posterior(
            adata_ref,
            sample_kwargs={
                'num_samples': 1000,
                'batch_size': BATCH_SIZE,
                'accelerator': 'gpu'
            }
        )

        mod.save(checkpoint_dir, overwrite=True)
        adata_checkpoint.write(f"{checkpoint_dir}/sc.h5ad")

        # Extract and save cell type signatures
        if 'means_per_cluster_mu_fg' in adata_checkpoint.varm.keys():
            inf_aver_ckpt = adata_checkpoint.varm['means_per_cluster_mu_fg'][
                [f'means_per_cluster_mu_fg_{i}' for i in adata_checkpoint.uns['mod']['factor_names']]
            ].copy()
        else:
            inf_aver_ckpt = adata_checkpoint.var[
                [f'means_per_cluster_mu_fg_{i}' for i in adata_checkpoint.uns['mod']['factor_names']]
            ].copy()
        inf_aver_ckpt.columns = adata_checkpoint.uns['mod']['factor_names']
        inf_aver_ckpt.to_csv(f"{checkpoint_dir}/cell_type_signatures.csv")

        cell_types_ckpt = list(inf_aver_ckpt.columns)
        pd.DataFrame({'cell_type': cell_types_ckpt}).to_csv(
            f"{checkpoint_dir}/cell_type_list.csv", index=False
        )

        print_progress(f"Checkpoint epoch {epochs_trained} saved: {checkpoint_dir}/")

        # Training curve
        plt.figure(figsize=(8, 5))
        history = mod.history['elbo_train']
        plt.plot(range(1, len(history) + 1), history.values, 'b-', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (ELBO)', fontsize=12)
        plt.title(f'Training History (Epoch {epochs_trained})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{checkpoint_dir}/training_history.png", dpi=150, bbox_inches='tight')
        plt.close()

    train_time = time.time() - train_start
    print_progress(f"Training done! Time: {train_time:.1f}s ({train_time/60:.1f}min)")

    # ============================================================
    # Step 5: Export final results
    # ============================================================

    print_step("Exporting final results", 5, 5)

    final_dir = f"{results_folder}/final"
    os.makedirs(final_dir, exist_ok=True)

    print_progress("Exporting posterior...")
    export_start = time.time()
    adata_ref = mod.export_posterior(
        adata_ref,
        sample_kwargs={
            'num_samples': 1000,
            'batch_size': BATCH_SIZE,
            'accelerator': 'gpu'
        }
    )
    print_progress(f"Posterior export done! Time: {time.time() - export_start:.1f}s")

    print_progress("Saving model...")
    mod.save(final_dir, overwrite=True)
    print_progress(f"Model saved to: {final_dir}/")

    print_progress("Saving AnnData...")
    save_start = time.time()
    adata_ref.write(f"{final_dir}/sc.h5ad")
    print_progress(f"AnnData saved! Time: {time.time() - save_start:.1f}s")

    print_progress("Extracting cell type signatures...")
    if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():
        inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][
            [f'means_per_cluster_mu_fg_{i}' for i in adata_ref.uns['mod']['factor_names']]
        ].copy()
    else:
        inf_aver = adata_ref.var[
            [f'means_per_cluster_mu_fg_{i}' for i in adata_ref.uns['mod']['factor_names']]
        ].copy()

    inf_aver.columns = adata_ref.uns['mod']['factor_names']
    inf_aver.to_csv(f"{final_dir}/cell_type_signatures.csv")
    print_progress(f"Cell type signatures saved: {final_dir}/cell_type_signatures.csv")

    cell_types = list(inf_aver.columns)
    pd.DataFrame({'cell_type': cell_types}).to_csv(
        f"{final_dir}/cell_type_list.csv", index=False
    )

    # Final training curve
    plt.figure(figsize=(10, 6))
    history = mod.history['elbo_train']
    plt.plot(range(1, len(history) + 1), history.values, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (ELBO)', fontsize=12)
    plt.title(f'{args.tissue.capitalize()} SC Regression Model Training', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{final_dir}/training_history.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Summary
    total_time = time.time() - START_TIME
    total_time_str = str(timedelta(seconds=int(total_time)))

    print("\n" + "="*60)
    print(f"Training complete! Total time: {total_time_str}")
    print("="*60)
    print(f"\nResults saved to: {results_folder}/")
    print(f"\nSignature matrix: {inf_aver.shape[0]:,} genes x {inf_aver.shape[1]} cell types")
    print(f"Cell types ({len(cell_types)}):")
    for i, ct in enumerate(cell_types, 1):
        print(f"  {i:2d}. {ct}")


if __name__ == "__main__":
    main()
