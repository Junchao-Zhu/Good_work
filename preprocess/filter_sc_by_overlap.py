#!/usr/bin/env python
"""Filter single-cell h5ad files to keep only genes that overlap with ST HVGs."""

import numpy as np
import scanpy as sc
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def filter_sc_data(input_dir, output_dir, overlap_genes, tissue_name):
    """
    Filter single-cell data to keep only overlapping genes.

    Parameters:
    -----------
    input_dir : Path
        Original SC data directory
    output_dir : Path
        Output directory
    overlap_genes : np.ndarray
        Overlapping gene list
    tissue_name : str
        Tissue name
    """
    logging.info(f"\n{'='*70}")
    logging.info(f"Processing {tissue_name} single-cell data")
    logging.info("="*70)

    overlap_genes_set = set(overlap_genes)
    logging.info(f"Overlap genes: {len(overlap_genes)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    h5ad_files = sorted(input_dir.glob("*.h5ad"))
    logging.info(f"Found {len(h5ad_files)} .h5ad files")

    filtered_count = 0
    skipped_count = 0

    for h5ad_file in tqdm(h5ad_files, desc=f"Processing {tissue_name}"):
        try:
            adata = sc.read_h5ad(h5ad_file)
            original_n_genes = adata.n_vars
            original_n_cells = adata.n_obs

            genes_in_data = adata.var_names.tolist()
            genes_to_keep = [gene for gene in overlap_genes if gene in genes_in_data]

            if len(genes_to_keep) == 0:
                logging.warning(f"  {h5ad_file.name}: no overlap genes, skipped")
                skipped_count += 1
                continue

            adata_filtered = adata[:, genes_to_keep].copy()

            output_file = output_dir / h5ad_file.name
            adata_filtered.write_h5ad(output_file)

            logging.info(f"  {h5ad_file.name}")
            logging.info(f"      Original: {original_n_cells} cells x {original_n_genes} genes")
            logging.info(f"      Filtered: {adata_filtered.n_obs} cells x {adata_filtered.n_vars} genes")
            logging.info(f"      Gene retention: {adata_filtered.n_vars}/{len(overlap_genes)} ({adata_filtered.n_vars/len(overlap_genes)*100:.1f}%)")

            filtered_count += 1

        except Exception as e:
            logging.error(f"  Error processing {h5ad_file.name}: {e}")
            skipped_count += 1

    logging.info(f"\nProcessed: {filtered_count} files")
    logging.info(f"Skipped: {skipped_count} files")

    if filtered_count > 0:
        first_output = output_dir / h5ad_files[0].name
        verify_data = sc.read_h5ad(first_output)
        logging.info(f"\nVerification ({h5ad_files[0].name}):")
        logging.info(f"  Filtered genes: {verify_data.n_vars}")
        logging.info(f"  Filtered cells: {verify_data.n_obs}")
        logging.info(f"  First 5 genes: {verify_data.var_names[:5].tolist()}")


def main():
    logging.info("="*70)
    logging.info("Filter single-cell data by gene overlap")
    logging.info("="*70)

    SCRIPT_DIR = Path(__file__).resolve().parent
    gene_overlap_dir = SCRIPT_DIR / "gene_overlap"
    sc_data_dir = SCRIPT_DIR / "sc_data_gene_mapped"

    datasets = [
        {
            'name': 'Breast SC (BC overlap)',
            'overlap_file': 'BC_hvg_in_breast_sc.npy',
            'input_dir': 'breast_sc_data',
            'output_dir': 'breast_sc_BC_filtered'
        },
        {
            'name': 'Breast SC (HER2 overlap)',
            'overlap_file': 'HER2_hvg_in_breast_sc.npy',
            'input_dir': 'breast_sc_data',
            'output_dir': 'breast_sc_HER2_filtered'
        },
        {
            'name': 'Kidney SC',
            'overlap_file': 'Kidney_hvg_in_kidney_sc.npy',
            'input_dir': 'kidney_sc_data',
            'output_dir': 'kidney_sc_filtered'
        }
    ]

    for config in datasets:
        overlap_path = gene_overlap_dir / config['overlap_file']

        if not overlap_path.exists():
            logging.warning(f"\n  Skipping {config['name']}: overlap file not found ({config['overlap_file']})")
            continue

        overlap_genes = np.load(overlap_path, allow_pickle=True)
        logging.info(f"\nLoaded {len(overlap_genes)} overlap genes from {config['overlap_file']}")

        input_dir = sc_data_dir / config['input_dir']
        output_dir = SCRIPT_DIR / "filter_sc_expression" / config['output_dir']

        if not input_dir.exists():
            logging.warning(f"  Input directory not found: {input_dir}")
            continue

        filter_sc_data(
            input_dir=input_dir,
            output_dir=output_dir,
            overlap_genes=overlap_genes,
            tissue_name=config['name']
        )

    logging.info("\n" + "="*70)
    logging.info("All data processed!")
    logging.info("="*70)

    output_base = SCRIPT_DIR / "filter_sc_expression"
    if output_base.exists():
        logging.info("\nOutput directory structure:")
        for subdir in sorted(output_base.iterdir()):
            if subdir.is_dir():
                file_count = len(list(subdir.glob("*.h5ad")))
                logging.info(f"  {subdir.name}/: {file_count} files")


if __name__ == "__main__":
    main()
