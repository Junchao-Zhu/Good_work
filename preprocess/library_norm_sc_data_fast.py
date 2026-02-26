#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Apply Log1p transformation to single-cell data (no Library-size Normalization).
Supports: GPU acceleration / multiprocessing parallelism.

Usage:
    python library_norm_sc_data_fast.py --src ./raw_data/sc_data --dst ./sc_data
"""

import os
import warnings
from pathlib import Path
import scanpy as sc
import logging
import argparse
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm
from scipy.sparse import issparse

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================
#  Default Config
# ============================
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SRC_BASE = Path(os.environ.get("SC_RAW_DATA", SCRIPT_DIR / "raw_data" / "sc_data"))
DST_BASE = SCRIPT_DIR / "sc_data"

DATASETS = ["breast_sc_data", "kidney_sc_data"]


# ============================
#  GPU acceleration
# ============================

def log1p_with_gpu(file_path, output_path):
    """Log1p transformation using GPU acceleration"""
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cp_sparse
        from scipy.sparse import csr_matrix

        adata = sc.read_h5ad(str(file_path))

        is_sparse = issparse(adata.X)

        X = adata.X
        if is_sparse:
            X_gpu = cp_sparse.csr_matrix(X)
            X_dense = X_gpu.toarray()
            X_log = cp.log1p(X_dense)
            adata.X = csr_matrix(cp.asnumpy(X_log))
        else:
            X_gpu = cp.array(X)
            X_log = cp.log1p(X_gpu)
            adata.X = cp.asnumpy(X_log)

        output_path.parent.mkdir(exist_ok=True, parents=True)
        adata.write_h5ad(output_path)

        return {
            'success': True,
            'file': file_path.name,
            'n_cells': adata.n_obs,
            'n_genes': adata.n_vars,
            'is_sparse': is_sparse,
            'dtype': str(adata.X.dtype),
            'mode': 'GPU'
        }

    except ImportError:
        return None  # Fallback to CPU
    except Exception as e:
        return {
            'success': False,
            'file': file_path.name,
            'error': str(e),
            'mode': 'GPU'
        }


# ============================
#  CPU functions
# ============================

def log1p_with_cpu(file_path, output_path):
    """Log1p transformation using CPU"""
    try:
        adata = sc.read_h5ad(str(file_path))
        is_sparse = issparse(adata.X)

        # Only log1p, no normalize_total
        sc.pp.log1p(adata)

        output_path.parent.mkdir(exist_ok=True, parents=True)
        adata.write_h5ad(output_path)

        return {
            'success': True,
            'file': file_path.name,
            'n_cells': adata.n_obs,
            'n_genes': adata.n_vars,
            'is_sparse': is_sparse,
            'dtype': str(adata.X.dtype),
            'mode': 'CPU'
        }

    except Exception as e:
        return {
            'success': False,
            'file': file_path.name,
            'error': str(e),
            'mode': 'CPU'
        }


def process_file_wrapper(args):
    """Multiprocessing wrapper"""
    file_path, output_path, use_gpu = args
    if use_gpu:
        result = log1p_with_gpu(file_path, output_path)
        if result is None:
            result = log1p_with_cpu(file_path, output_path)
    else:
        result = log1p_with_cpu(file_path, output_path)
    return result


# ============================
#  Main
# ============================

def main(use_gpu=True, n_jobs=4, src_base=None, dst_base=None):
    src = Path(src_base) if src_base else SRC_BASE
    dst = Path(dst_base) if dst_base else DST_BASE

    print("="*60)
    print("Log1p Only (No Library Normalization)")
    print(f"Input:  {src}")
    print(f"Output: {dst}")
    print(f"GPU: {'ON' if use_gpu else 'OFF'}")
    if not use_gpu:
        print(f"Parallel jobs: {n_jobs}")
    print("="*60)

    all_tasks = []
    for dataset in DATASETS:
        src_dir = src / dataset
        dst_dir = dst / dataset

        if not src_dir.exists():
            print(f"[WARN] Source directory not found: {src_dir}")
            continue

        h5ad_files = sorted(src_dir.glob("*.h5ad"))
        print(f"\nDataset {dataset}: {len(h5ad_files)} files")

        for file in h5ad_files:
            output_path = dst_dir / file.name
            all_tasks.append((file, output_path, use_gpu))

    print(f"\nTotal: {len(all_tasks)} files to process\n")

    if len(all_tasks) == 0:
        print("No files to process!")
        return

    results = []
    if use_gpu:
        for task in tqdm(all_tasks, desc="Processing (GPU)", unit="file"):
            results.append(process_file_wrapper(task))
    else:
        with Pool(n_jobs) as pool:
            for result in tqdm(pool.imap(process_file_wrapper, all_tasks),
                              total=len(all_tasks),
                              desc="Processing (CPU)",
                              unit="file"):
                results.append(result)

    success_results = [r for r in results if r and r.get('success', False)]
    failed_results = [r for r in results if r and not r.get('success', True)]

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Success: {len(success_results)}/{len(all_tasks)}")
    print(f"Failed:  {len(failed_results)}/{len(all_tasks)}")

    if success_results:
        print("\n" + "-"*60)
        print("OUTPUT DATA FORMAT")
        print("-"*60)

        total_cells = sum(r['n_cells'] for r in success_results)
        total_genes = set(r['n_genes'] for r in success_results)
        dtypes = set(r['dtype'] for r in success_results)
        sparse_count = sum(1 for r in success_results if r['is_sparse'])

        print(f"Total cells:    {total_cells:,}")
        print(f"Genes per file: {total_genes}")
        print(f"Data type:      {dtypes}")
        print(f"Sparse matrix:  {sparse_count}/{len(success_results)} files")

        print("\n" + "-"*60)
        print("FILE DETAILS")
        print("-"*60)
        print(f"{'File':<40} {'Cells':>10} {'Genes':>8} {'Sparse':>8} {'Mode':>6}")
        print("-"*60)
        for r in success_results:
            sparse_str = "Yes" if r['is_sparse'] else "No"
            print(f"{r['file']:<40} {r['n_cells']:>10,} {r['n_genes']:>8} {sparse_str:>8} {r['mode']:>6}")

    if failed_results:
        print("\n" + "-"*60)
        print("FAILED FILES")
        print("-"*60)
        for r in failed_results:
            print(f"  {r['file']}: {r.get('error', 'Unknown error')}")

    print("\n" + "="*60)
    print(f"Output directory: {dst}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Log1p Only (No Library Norm)')
    parser.add_argument('--mode', type=str, default='gpu',
                        choices=['gpu', 'cpu'],
                        help='gpu or cpu')
    parser.add_argument('--n-jobs', type=int, default=8,
                        help='Parallel jobs for CPU mode')
    parser.add_argument('--src', type=str, default=None,
                        help='Source directory for raw SC data')
    parser.add_argument('--dst', type=str, default=None,
                        help='Destination directory for output')

    args = parser.parse_args()
    main(use_gpu=(args.mode == 'gpu'), n_jobs=args.n_jobs,
         src_base=args.src, dst_base=args.dst)
