# -*- coding: utf-8 -*-
"""
Select top-K HVG subset from existing hvg2000 data.

No pybiomart / reloading original h5ad needed. Directly selects top-K by
variance from the hvg2000 h5ad files.

Usage:
    python prepare_hvg_subset.py --hvg_k 50
    python prepare_hvg_subset.py --hvg_k 150
    python prepare_hvg_subset.py --hvg_k 300
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HVG2000_BASE = os.path.join(SCRIPT_DIR, "hvg2000_data")
OUTPUT_BASE_TEMPLATE = os.path.join(SCRIPT_DIR, "hvg{}_data")

DATASETS = {
    "BC": [f"SPA{i}" for i in range(51, 119)],
    "HER2": [f"SPA{i}" for i in range(119, 155)],
    "Kidney": [f"NCBI{i}" for i in range(692, 715)],
}


def process_dataset(dataset_name, slides, hvg_k):
    """Select top-K HVGs from hvg2000 data."""
    hvg2000_dir = os.path.join(HVG2000_BASE, dataset_name)
    h5ad_dir = os.path.join(hvg2000_dir, "h5ad")
    sig_path = os.path.join(hvg2000_dir, "cell_type_signatures_hvg2000.csv")

    output_dir = os.path.join(OUTPUT_BASE_TEMPLATE.format(hvg_k), dataset_name)
    out_h5ad_dir = os.path.join(output_dir, "h5ad")
    os.makedirs(out_h5ad_dir, exist_ok=True)

    # Load signature matrix (2000 genes x n_cell_types)
    sig_df = pd.read_csv(sig_path, index_col=0)
    hvg2000_genes = list(sig_df.index)
    logging.info(f"[{dataset_name}] Signature matrix: {sig_df.shape[0]} genes x {sig_df.shape[1]} cell types")

    # Load all slides and compute gene variance
    logging.info(f"[{dataset_name}] Loading {len(slides)} slides for gene variance...")
    all_expr = []
    valid_slides = []
    for slide_id in slides:
        path = os.path.join(h5ad_dir, f"{slide_id}.h5ad")
        if not os.path.exists(path):
            logging.warning(f"  {slide_id} not found, skipped.")
            continue
        adata = sc.read_h5ad(path)
        X = adata.X.toarray() if issparse(adata.X) else adata.X
        all_expr.append(X)
        valid_slides.append(slide_id)

    merged_expr = np.vstack(all_expr)
    logging.info(f"[{dataset_name}] Merged: {merged_expr.shape[0]} spots x {merged_expr.shape[1]} genes")

    # Variance in log1p space (consistent with prepare_st_for_deconv.py)
    log_expr = np.log1p(merged_expr)
    gene_vars = log_expr.var(axis=0)

    # Select top-K
    topk_idx = np.argsort(gene_vars)[::-1][:hvg_k]
    hvg_genes = [hvg2000_genes[i] for i in sorted(topk_idx)]
    logging.info(f"[{dataset_name}] Selected top-{hvg_k} HVGs (from 2000)")

    # Save HVG gene list
    pd.DataFrame({"gene_id": hvg_genes}).to_csv(
        os.path.join(output_dir, "hvg_genes.csv"), index=False
    )

    # Save subset signature matrix
    sig_subset = sig_df.loc[hvg_genes, :].copy()
    sig_subset.to_csv(os.path.join(output_dir, f"cell_type_signatures_hvg{hvg_k}.csv"))
    logging.info(f"[{dataset_name}] Signature subset: {sig_subset.shape}")

    # Save per-slide h5ad
    for slide_id in valid_slides:
        src_path = os.path.join(h5ad_dir, f"{slide_id}.h5ad")
        adata = sc.read_h5ad(src_path)
        adata_sub = adata[:, hvg_genes].copy()
        adata_sub.write(os.path.join(out_h5ad_dir, f"{slide_id}.h5ad"))

    logging.info(f"[{dataset_name}] Done: {len(valid_slides)} slides -> {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Select top-K HVG subset from hvg2000 data")
    parser.add_argument("--hvg_k", type=int, required=True,
                        help="Number of HVGs to select (e.g. 50, 150, 300)")
    args = parser.parse_args()

    if args.hvg_k >= 2000:
        logging.error("hvg_k must be < 2000 (hvg2000 already exists)")
        return

    logging.info(f"Preparing HVG-{args.hvg_k} data subset")
    logging.info(f"Output: {OUTPUT_BASE_TEMPLATE.format(args.hvg_k)}/")

    for name, slides in DATASETS.items():
        process_dataset(name, slides, args.hvg_k)

    logging.info("\nAll done!")


if __name__ == "__main__":
    main()
