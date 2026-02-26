#!/usr/bin/env python
"""Filter HVG expression data to keep only genes that overlap with SC data."""

import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def filter_hvg_data(input_dir, output_dir, overlap_genes, tissue_name):
    """
    Filter HVG data to keep only overlapping genes.

    Parameters:
    -----------
    input_dir : Path
        Original HVG data directory
    output_dir : Path
        Output directory
    overlap_genes : np.ndarray
        Overlapping gene list
    tissue_name : str
        Tissue name
    """
    logging.info(f"\n{'='*70}")
    logging.info(f"Processing {tissue_name} data")
    logging.info("="*70)

    overlap_genes_set = set(overlap_genes)
    logging.info(f"Overlap genes: {len(overlap_genes)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(input_dir.glob("*.npy"))
    logging.info(f"Found {len(npy_files)} .npy files")

    filtered_count = 0
    skipped_count = 0

    for npy_file in tqdm(npy_files, desc=f"Processing {tissue_name}"):
        try:
            data = np.load(npy_file, allow_pickle=True).item()
            gene_names = data['gene_names']
            expression = data['expression']

            # Find gene indices in overlap order
            gene_indices = []
            filtered_gene_names = []

            for gene in overlap_genes:
                if gene in gene_names:
                    idx = np.where(gene_names == gene)[0]
                    if len(idx) > 0:
                        gene_indices.append(idx[0])
                        filtered_gene_names.append(gene)

            if len(gene_indices) == 0:
                logging.warning(f"  {npy_file.name}: no overlap genes, skipped")
                skipped_count += 1
                continue

            filtered_expression = expression[:, gene_indices]

            filtered_data = {
                'gene_names': np.array(filtered_gene_names),
                'expression': filtered_expression
            }

            output_file = output_dir / npy_file.name
            np.save(output_file, filtered_data)

            filtered_count += 1

        except Exception as e:
            logging.error(f"  Error processing {npy_file.name}: {e}")
            skipped_count += 1

    logging.info(f"\nProcessed: {filtered_count} files")
    logging.info(f"Skipped: {skipped_count} files")

    if filtered_count > 0:
        first_output = output_dir / npy_files[0].name
        verify_data = np.load(first_output, allow_pickle=True).item()
        logging.info(f"\nVerification ({npy_files[0].name}):")
        logging.info(f"  Original genes: 1000")
        logging.info(f"  Filtered genes: {len(verify_data['gene_names'])}")
        logging.info(f"  Expression shape: {verify_data['expression'].shape}")
        logging.info(f"  First 5 genes: {verify_data['gene_names'][:5].tolist()}")


def main():
    logging.info("="*70)
    logging.info("Filter HVG data by gene overlap")
    logging.info("="*70)

    SCRIPT_DIR = Path(__file__).resolve().parent
    gene_overlap_dir = SCRIPT_DIR / "gene_overlap"
    hvg_expr_dir = SCRIPT_DIR / "hvg_expr_npy"
    output_base_dir = SCRIPT_DIR / "filter_hvg_expression"

    datasets = {
        'BC_hvg_in_breast_sc.npy': {
            'name': 'BC (Breast Cancer)',
            'hvg_dir': 'BC'
        },
        'HER2_hvg_in_breast_sc.npy': {
            'name': 'HER2',
            'hvg_dir': 'HER2'
        },
        'Kidney_hvg_in_kidney_sc.npy': {
            'name': 'Kidney',
            'hvg_dir': 'Kidney'
        }
    }

    for overlap_file, config in datasets.items():
        overlap_path = gene_overlap_dir / overlap_file

        if not overlap_path.exists():
            logging.warning(f"\n  Skipping {config['name']}: overlap file not found ({overlap_file})")
            continue

        overlap_genes = np.load(overlap_path, allow_pickle=True)
        logging.info(f"\nLoaded {len(overlap_genes)} overlap genes from {overlap_file}")

        input_dir = hvg_expr_dir / config['hvg_dir']
        output_dir = output_base_dir / config['hvg_dir']

        if not input_dir.exists():
            logging.warning(f"  Input directory not found: {input_dir}")
            continue

        filter_hvg_data(
            input_dir=input_dir,
            output_dir=output_dir,
            overlap_genes=overlap_genes,
            tissue_name=config['name']
        )

    logging.info("\n" + "="*70)
    logging.info("All data processed!")
    logging.info("="*70)

    logging.info("\nOutput directory structure:")
    logging.info(f"Output: {output_base_dir}")
    if output_base_dir.exists():
        for subdir in sorted(output_base_dir.iterdir()):
            if subdir.is_dir():
                file_count = len(list(subdir.glob("*.npy")))
                logging.info(f"  {subdir.name}/: {file_count} files")


if __name__ == "__main__":
    main()
