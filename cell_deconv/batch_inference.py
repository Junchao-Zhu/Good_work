# -*- coding: utf-8 -*-
"""
Cell2location batch deconvolution (multi-GPU parallel).

Usage:
    # Single GPU, all slides
    python batch_inference.py

    # Multi-GPU parallel (run in separate terminals)
    python batch_inference.py --gpu 0 --total_gpus 8
    python batch_inference.py --gpu 1 --total_gpus 8
    ...
"""

import os
import sys
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import issparse

# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Cell2location hyperparameters
N_CELLS_PER_LOCATION = 30
DETECTION_ALPHA = 20
MAX_EPOCHS = 10000


def get_datasets_config(hvg_k):
    """Generate dataset config based on hvg_k."""
    data_base = os.path.join(SCRIPT_DIR, f"hvg{hvg_k}_data")
    return {
        "BC": {
            "h5ad_dir": f"{data_base}/BC/h5ad",
            "signature": f"{data_base}/BC/cell_type_signatures_hvg{hvg_k}.csv",
            "slides": [f"SPA{i}" for i in range(51, 119)]
        },
        "HER2": {
            "h5ad_dir": f"{data_base}/HER2/h5ad",
            "signature": f"{data_base}/HER2/cell_type_signatures_hvg{hvg_k}.csv",
            "slides": [f"SPA{i}" for i in range(119, 155)]
        },
        "Kidney": {
            "h5ad_dir": f"{data_base}/Kidney/h5ad",
            "signature": f"{data_base}/Kidney/cell_type_signatures_hvg{hvg_k}.csv",
            "slides": [f"NCBI{i}" for i in range(692, 715)]
        }
    }


def get_all_tasks(hvg_k):
    """Get all deconvolution tasks."""
    datasets = get_datasets_config(hvg_k)
    output_base = os.path.join(SCRIPT_DIR, f"deconv_results_hvg{hvg_k}")
    tasks = []
    for dataset_name, config in datasets.items():
        for slide_id in config['slides']:
            h5ad_path = os.path.join(config['h5ad_dir'], f"{slide_id}.h5ad")
            if os.path.exists(h5ad_path):
                tasks.append({
                    'dataset': dataset_name,
                    'slide_id': slide_id,
                    'h5ad_path': h5ad_path,
                    'signature': config['signature'],
                    'output_dir': os.path.join(output_base, dataset_name)
                })
    return tasks


def run_deconv_single(task):
    """Run deconvolution on a single slide."""
    slide_id = task['slide_id']
    h5ad_path = task['h5ad_path']
    signature_path = task['signature']
    output_dir = task['output_dir']

    print("\n" + "="*60)
    print(f"Processing: {task['dataset']}/{slide_id}")
    print("="*60)

    slide_output = os.path.join(output_dir, slide_id)

    # Skip if already processed
    if os.path.exists(os.path.join(slide_output, "cell_proportion.csv")):
        print(f"{slide_id} already processed, skipping")
        return True

    os.makedirs(slide_output, exist_ok=True)

    # Load signature matrix
    inf_aver = pd.read_csv(signature_path, index_col=0)
    print(f"Signature matrix: {inf_aver.shape[0]} genes x {inf_aver.shape[1]} cell types")

    # Load ST data
    adata_vis = sc.read_h5ad(h5ad_path)
    print(f"ST data: {adata_vis.shape[0]} spots x {adata_vis.shape[1]} genes")

    # Gene matching
    intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
    if len(intersect) != adata_vis.shape[1]:
        adata_vis = adata_vis[:, intersect].copy()
        inf_aver = inf_aver.loc[intersect, :].copy()

    if 'sample' not in adata_vis.obs.columns:
        adata_vis.obs['sample'] = slide_id

    # Import cell2location (after GPU setup)
    import cell2location
    from matplotlib import rcParams
    import matplotlib.pyplot as plt
    rcParams['pdf.fonttype'] = 42

    # Setup AnnData
    cell2location.models.Cell2location.setup_anndata(
        adata=adata_vis,
        batch_key="sample"
    )

    # Create model
    mod = cell2location.models.Cell2location(
        adata_vis,
        cell_state_df=inf_aver,
        N_cells_per_location=N_CELLS_PER_LOCATION,
        detection_alpha=DETECTION_ALPHA
    )

    # Train
    print(f"Training (max_epochs={MAX_EPOCHS})...", flush=True)
    mod.train(
        max_epochs=MAX_EPOCHS,
        batch_size=None,
        train_size=1,
        accelerator='gpu'
    )

    # Save training curve
    plt.figure(figsize=(8, 5))
    history = mod.history['elbo_train']
    plt.plot(history.index, history.values, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (ELBO)', fontsize=12)
    plt.title(f'{slide_id} Training', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{slide_output}/training_history.png", dpi=150)
    plt.close()

    # Export posterior
    adata_vis = mod.export_posterior(
        adata_vis,
        sample_kwargs={
            'num_samples': 1000,
            'batch_size': mod.adata.n_obs,
            'accelerator': 'gpu'
        }
    )

    # Extract cell abundance
    cell_type_names = adata_vis.uns['mod']['factor_names']

    def extract_abundance(obsm_data, obs_names, cell_types):
        if isinstance(obsm_data, pd.DataFrame):
            df = obsm_data.copy()
            df.columns = cell_types
            return df
        else:
            return pd.DataFrame(obsm_data, index=obs_names, columns=cell_types)

    cell_abundance_q05 = extract_abundance(
        adata_vis.obsm['q05_cell_abundance_w_sf'],
        adata_vis.obs_names,
        cell_type_names
    )

    cell_proportion = cell_abundance_q05.div(cell_abundance_q05.sum(axis=1), axis=0)

    # Save results
    cell_proportion.to_csv(f"{slide_output}/cell_proportion.csv")
    cell_abundance_q05.to_csv(f"{slide_output}/cell_abundance_q05.csv")
    pd.DataFrame({'cell_type': cell_type_names}).to_csv(
        f"{slide_output}/cell_type_list.csv", index=False
    )

    print(f"Done: {slide_output}/cell_proportion.csv")
    return True


def main():
    parser = argparse.ArgumentParser(description='Cell2location batch deconvolution')
    parser.add_argument('--gpu', type=int, default=0, help='Task allocation index')
    parser.add_argument('--total_gpus', type=int, default=1, help='Total GPU count')
    parser.add_argument('--cuda_device', type=int, default=None,
                        help='Actual CUDA device ID (defaults to --gpu)')
    parser.add_argument('--hvg_k', type=int, default=2000,
                        help='Number of HVGs (default: 2000)')
    args = parser.parse_args()

    # GPU setup: use external CUDA_VISIBLE_DEVICES
    cuda_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ["THEANO_FLAGS"] = 'device=cuda,floatX=float32,force_device=True'

    import torch
    torch.set_float32_matmul_precision('high')

    print(f'Using GPU {cuda_id}, HVG_K={args.hvg_k}')

    # Get all tasks
    all_tasks = get_all_tasks(args.hvg_k)
    print(f"Total tasks: {len(all_tasks)}")

    # Distribute tasks
    my_tasks = [t for i, t in enumerate(all_tasks) if i % args.total_gpus == args.gpu]
    print(f"GPU {args.gpu} handles {len(my_tasks)} tasks")

    # Execute tasks
    total_time = 0
    for i, task in enumerate(my_tasks):
        print(f"\n[{i+1}/{len(my_tasks)}] {task['dataset']}/{task['slide_id']}  "
              f"(GPU {cuda_id}, hvg{args.hvg_k})")
        t0 = time.time()
        try:
            run_deconv_single(task)
        except Exception as e:
            print(f"Error: {task['slide_id']}: {e}")
            continue
        elapsed = time.time() - t0
        total_time += elapsed
        avg = total_time / (i + 1)
        remaining = avg * (len(my_tasks) - i - 1)
        print(f"  Time {elapsed/60:.1f}min | Avg {avg/60:.1f}min/slide | "
              f"Remaining ~{remaining/60:.0f}min ({len(my_tasks)-i-1} slides)")

    print(f"\nGPU {cuda_id} all done! Total time {total_time/60:.1f}min")


if __name__ == "__main__":
    main()
