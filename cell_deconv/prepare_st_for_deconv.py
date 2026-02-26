# -*- coding: utf-8 -*-
"""
Prepare ST data for Cell2location deconvolution.

Steps:
1. Load ST h5ad files
2. Convert gene names to Ensembl IDs (if needed)
3. Intersect with signature matrix genes
4. Select top-K HVGs
5. Save npy and h5ad formats

Datasets:
- BC: SPA51-SPA118 (68 slides) - already ENSG format
- HER2: SPA119-SPA154 (36 slides) - Symbol format, needs pybiomart conversion
- Kidney: NCBI692-NCBI714 (23 slides) - Symbol format, uses var['gene_ids'] column

Usage:
    python prepare_st_for_deconv.py
    python prepare_st_for_deconv.py --hvg_k 1000
"""

import os
import re
import argparse
import logging
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse, csr_matrix
from pybiomart import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ST_PATH = os.environ.get("ST_DATA_PATH", os.path.join(SCRIPT_DIR, "st_data"))
PRETRAINED_DIR = os.path.join(SCRIPT_DIR, "pretrained")

# Dataset configuration
# convert_mode: "none" | "pybiomart" | "gene_ids_col"
DATASETS = {
    "BC": {
        "slides": [f"SPA{i}" for i in range(51, 119)],
        "signature": f"{PRETRAINED_DIR}/breast/final/cell_type_signatures.csv",
        "convert_mode": "none"  # already ENSG format
    },
    "HER2": {
        "slides": [f"SPA{i}" for i in range(119, 155)],
        "signature": f"{PRETRAINED_DIR}/breast/final/cell_type_signatures.csv",
        "convert_mode": "pybiomart"  # Symbol -> ENSG via pybiomart
    },
    "Kidney": {
        "slides": [f"NCBI{i}" for i in range(692, 715)],
        "signature": f"{PRETRAINED_DIR}/kidney/final/cell_type_signatures.csv",
        "convert_mode": "gene_ids_col"  # use var['gene_ids'] column
    }
}


# ============================================================
# Gene name conversion
# ============================================================

def load_gene_mapping():
    """Load Gene Symbol to ENSG mapping (for HER2)."""
    logging.info("Loading Ensembl gene annotations...")
    human_ds = Dataset(name="hsapiens_gene_ensembl", host="http://www.ensembl.org")
    hdf = human_ds.query(attributes=["ensembl_gene_id", "external_gene_name"]).dropna()

    sym2ens = {}
    for _, row in hdf.iterrows():
        sym = row["Gene name"].upper()
        ens = row["Gene stable ID"]
        if sym not in sym2ens:
            sym2ens[sym] = ens

    logging.info(f"Loaded {len(sym2ens)} gene mappings")
    return sym2ens


def clean_gene_name(g):
    return re.sub(r"\.\d+$", "", g).upper()


def convert_genes_pybiomart(adata, sym2ens):
    """Convert gene names from Symbol to ENSG using pybiomart."""
    original_genes = adata.var_names.tolist()
    new_genes = []
    valid_mask = []

    for g in original_genes:
        g_clean = clean_gene_name(g)
        if g_clean in sym2ens:
            new_genes.append(sym2ens[g_clean])
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    logging.info(f"pybiomart conversion: {len(original_genes)} -> {sum(valid_mask)} "
                 f"({sum(valid_mask)/len(original_genes)*100:.1f}%)")

    adata_new = adata[:, valid_mask].copy()
    adata_new.var_names = pd.Index(new_genes)
    adata_new.var_names_make_unique()
    return adata_new


def convert_genes_from_column(adata):
    """Use var['gene_ids'] column as ENSG gene names."""
    if 'gene_ids' not in adata.var.columns:
        raise ValueError("var does not have gene_ids column!")

    gene_ids = adata.var['gene_ids'].values
    valid_mask = pd.notnull(gene_ids) & (gene_ids != '')
    valid_mask = valid_mask & np.array([str(g).startswith('ENSG') for g in gene_ids])

    logging.info(f"gene_ids column conversion: {len(gene_ids)} -> {sum(valid_mask)} "
                 f"({sum(valid_mask)/len(gene_ids)*100:.1f}%)")

    adata_new = adata[:, valid_mask].copy()
    adata_new.var_names = pd.Index(gene_ids[valid_mask])
    adata_new.var_names_make_unique()
    return adata_new


def convert_genes(adata, convert_mode, sym2ens):
    if convert_mode == "none":
        return adata
    elif convert_mode == "pybiomart":
        return convert_genes_pybiomart(adata, sym2ens)
    elif convert_mode == "gene_ids_col":
        return convert_genes_from_column(adata)
    else:
        raise ValueError(f"Unknown convert_mode: {convert_mode}")


# ============================================================
# HVG selection and data saving
# ============================================================

def load_signature_genes(sig_path):
    sig = pd.read_csv(sig_path, index_col=0)
    logging.info(f"Signature matrix: {sig.shape[0]} genes x {sig.shape[1]} cell types")
    return set(sig.index), sig


def get_common_genes(slide_list, st_path, sig_genes, convert_mode, sym2ens):
    """Find common genes across all slides and signature matrix."""
    common_genes = None
    valid_slides = []

    for slide_id in slide_list:
        path = os.path.join(st_path, f"{slide_id}.h5ad")
        if not os.path.exists(path):
            logging.warning(f"{slide_id} not found, skipped.")
            continue

        adata = sc.read_h5ad(path)
        adata = convert_genes(adata, convert_mode, sym2ens)

        st_genes = set(adata.var_names)
        intersect = st_genes & sig_genes

        if common_genes is None:
            common_genes = intersect
        else:
            common_genes &= intersect

        valid_slides.append(slide_id)
        logging.info(f"{slide_id}: ST genes={len(st_genes)}, common={len(common_genes)}")

    return sorted(common_genes) if common_genes else [], valid_slides


def select_hvg(slide_list, st_path, common_genes, hvg_k, convert_mode, sym2ens):
    """Select top-K HVGs from common genes."""
    all_expr = []

    for slide_id in slide_list:
        path = os.path.join(st_path, f"{slide_id}.h5ad")
        if not os.path.exists(path):
            continue

        adata = sc.read_h5ad(path)
        adata = convert_genes(adata, convert_mode, sym2ens)

        common_in_adata = [g for g in common_genes if g in adata.var_names]
        adata = adata[:, common_in_adata].copy()

        X = adata.X.toarray() if issparse(adata.X) else adata.X
        all_expr.append(X)

    merged_expr = np.vstack(all_expr)
    logging.info(f"Merged expression matrix: {merged_expr.shape}")

    # Compute variance in log1p space
    log_expr = np.log1p(merged_expr)
    gene_vars = log_expr.var(axis=0)

    topk_idx = np.argsort(gene_vars)[::-1][:hvg_k]
    hvg_genes = [common_genes[i] for i in topk_idx]

    logging.info(f"Selected {len(hvg_genes)} HVGs")
    return hvg_genes


def save_slide_data(slide_list, st_path, hvg_genes, output_dir, sig_df,
                    hvg_k, convert_mode, sym2ens):
    """Save per-slide data in npy and h5ad formats."""
    npy_dir = os.path.join(output_dir, "npy")
    h5ad_dir = os.path.join(output_dir, "h5ad")
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(h5ad_dir, exist_ok=True)

    # Save HVG gene list
    hvg_df = pd.DataFrame({"gene_id": hvg_genes})
    hvg_df.to_csv(os.path.join(output_dir, "hvg_genes.csv"), index=False)

    # Save subset signature matrix
    sig_subset = sig_df.loc[hvg_genes, :].copy()
    sig_subset.to_csv(os.path.join(output_dir, f"cell_type_signatures_hvg{hvg_k}.csv"))
    logging.info(f"Saved subset signature matrix: {sig_subset.shape}")

    for slide_id in slide_list:
        path = os.path.join(st_path, f"{slide_id}.h5ad")
        if not os.path.exists(path):
            continue

        adata_orig = sc.read_h5ad(path)
        adata_converted = convert_genes(adata_orig, convert_mode, sym2ens)

        hvg_in_adata = [g for g in hvg_genes if g in adata_converted.var_names]
        if len(hvg_in_adata) != len(hvg_genes):
            logging.warning(f"{slide_id}: only {len(hvg_in_adata)}/{len(hvg_genes)} HVG genes, skipped")
            continue

        adata = adata_converted[:, hvg_genes].copy()
        X = adata.X.toarray() if issparse(adata.X) else adata.X

        # Save npy format
        patch_data_list = []
        for i in range(X.shape[0]):
            patch_data = {
                "sample_id": slide_id,
                "spot_id": f"spot_{i}",
                "patch_id": f"patch_{i}",
                "gene_names": np.array(hvg_genes),
                "expression": X[i, :].astype(np.float32),
                "expression_log1p": np.log1p(X[i, :]).astype(np.float32),
            }
            patch_data_list.append(patch_data)

        np.save(os.path.join(npy_dir, f"{slide_id}.npy"),
                np.array(patch_data_list, dtype=object))

        # Save h5ad format
        adata_new = sc.AnnData(
            X=csr_matrix(X),
            obs=adata_orig.obs.copy(),
            var=pd.DataFrame(index=hvg_genes),
        )
        if "spatial" in adata_orig.uns:
            adata_new.uns["spatial"] = adata_orig.uns["spatial"]
        if "spatial" in adata_orig.obsm:
            adata_new.obsm["spatial"] = adata_orig.obsm["spatial"]
        adata_new.obs["sample"] = slide_id
        adata_new.write(os.path.join(h5ad_dir, f"{slide_id}.h5ad"))

        logging.info(f"[Saved] {slide_id}: {X.shape[0]} spots")


def process_dataset(dataset_name, config, hvg_k, output_base, sym2ens):
    """Process a single dataset."""
    logging.info("\n" + "="*60)
    logging.info(f"Processing dataset: {dataset_name}")
    logging.info(f"Gene conversion mode: {config['convert_mode']}")
    logging.info("="*60)

    output_dir = os.path.join(output_base, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    sig_genes, sig_df = load_signature_genes(config["signature"])

    common_genes, valid_slides = get_common_genes(
        config["slides"], ST_PATH, sig_genes,
        config["convert_mode"], sym2ens
    )
    logging.info(f"Common genes: {len(common_genes)}, valid slides: {len(valid_slides)}")

    if len(common_genes) < hvg_k:
        actual_k = len(common_genes)
        logging.warning(f"Common genes < {hvg_k}, using {actual_k}")
    else:
        actual_k = hvg_k

    hvg_genes = select_hvg(
        valid_slides, ST_PATH, common_genes, actual_k,
        config["convert_mode"], sym2ens
    )

    save_slide_data(
        valid_slides, ST_PATH, hvg_genes, output_dir, sig_df,
        hvg_k, config["convert_mode"], sym2ens
    )

    logging.info(f"{dataset_name} done!")


def main():
    parser = argparse.ArgumentParser(description="Prepare ST data for cell2location deconv")
    parser.add_argument("--hvg_k", type=int, default=2000,
                        help="Number of HVGs to select (default: 2000)")
    args = parser.parse_args()

    output_base = os.path.join(SCRIPT_DIR, f"hvg{args.hvg_k}_data")

    logging.info("="*60)
    logging.info("Preparing ST data for deconvolution")
    logging.info(f"HVG_K = {args.hvg_k}, output -> {output_base}")
    logging.info("="*60)

    # Load gene mapping (only needed for HER2)
    sym2ens = load_gene_mapping()

    for name, config in DATASETS.items():
        process_dataset(name, config, args.hvg_k, output_base, sym2ens)

    logging.info("\n" + "="*60)
    logging.info("All done!")
    logging.info("="*60)
    logging.info(f"\nOutput: {output_base}/")
    logging.info("  ├── BC/")
    logging.info("  ├── HER2/")
    logging.info("  └── Kidney/")


if __name__ == "__main__":
    main()
