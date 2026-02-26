#!/usr/bin/env python
"""
Convert gene names from Ensembl IDs to Gene Symbols in single-cell h5ad files.

Usage:
    python map_genes.py
"""

import re
import logging
import pandas as pd
import anndata
from pathlib import Path
from pybiomart import Dataset
from tqdm import tqdm
from scipy.sparse import issparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

clean_gene = lambda g: re.sub(r"\.\d+$", "", g).upper()
is_ensembl_id = lambda s: bool(re.fullmatch(r"ENSG\d{11}", s))

def load_ensembl_annotations():
    human_ds = Dataset(name="hsapiens_gene_ensembl", host="http://www.ensembl.org")
    hdf = human_ds.query(attributes=["ensembl_gene_id", "external_gene_name"]).dropna()
    ens2sym = dict(zip(hdf["Gene stable ID"], hdf["Gene name"]))
    valid_hs = set(hdf["Gene name"].str.upper())
    return ens2sym, valid_hs

def map_gene(g, ens2sym, valid_hs):
    g2 = clean_gene(g)
    if is_ensembl_id(g2):
        return ens2sym.get(g2)
    if g2 in valid_hs:
        return g2
    return None

def convert_file(path, out_path, ens2sym, valid_hs):
    """Convert gene names for a single file"""
    try:
        adata = anndata.read_h5ad(str(path))
        original_genes = len(adata.var_names)
        original_cells = adata.n_obs

        genes = adata.var_names.tolist()
        mapped = [map_gene(g, ens2sym, valid_hs) for g in genes]
        mask = pd.notnull(mapped)

        adata = adata[:, mask].copy()
        adata.var_names = [m for m in mapped if pd.notnull(m)]
        adata.var_names_make_unique()

        out_path.parent.mkdir(exist_ok=True, parents=True)
        adata.write_h5ad(out_path)

        return {
            'success': True,
            'file': path.name,
            'n_cells': adata.n_obs,
            'original_genes': original_genes,
            'mapped_genes': adata.n_vars,
            'is_sparse': issparse(adata.X),
            'dtype': str(adata.X.dtype)
        }
    except Exception as e:
        return {
            'success': False,
            'file': path.name,
            'error': str(e)
        }

def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    INPUT = SCRIPT_DIR / "sc_data"
    OUTPUT = SCRIPT_DIR / "sc_data_gene_mapped"

    print("="*60)
    print("Gene Name Mapping (Ensembl ID -> Gene Symbol)")
    print(f"Input:  {INPUT}")
    print(f"Output: {OUTPUT}")
    print("="*60)

    print("\nLoading Ensembl annotations...")
    ens2sym, valid_hs = load_ensembl_annotations()
    print(f"Ensembl mappings: {len(ens2sym):,}")
    print(f"Valid gene symbols: {len(valid_hs):,}")

    all_tasks = []
    for dataset in ["breast_sc_data", "kidney_sc_data"]:
        input_dir = INPUT / dataset
        if not input_dir.exists():
            print(f"[WARN] Directory not found: {input_dir}")
            continue
        files = sorted(input_dir.glob("*.h5ad"))
        print(f"\nDataset {dataset}: {len(files)} files")
        for f in files:
            all_tasks.append((f, OUTPUT / dataset / f.name))

    print(f"\nTotal: {len(all_tasks)} files to process\n")

    if len(all_tasks) == 0:
        print("No files to process!")
        return

    results = []
    for src, dst in tqdm(all_tasks, desc="Mapping genes", unit="file"):
        result = convert_file(src, dst, ens2sym, valid_hs)
        results.append(result)

    success_results = [r for r in results if r.get('success', False)]
    failed_results = [r for r in results if not r.get('success', True)]

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
        total_original = sum(r['original_genes'] for r in success_results)
        total_mapped = sum(r['mapped_genes'] for r in success_results)
        mapped_genes_set = set(r['mapped_genes'] for r in success_results)
        dtypes = set(r['dtype'] for r in success_results)

        print(f"Total cells:      {total_cells:,}")
        print(f"Genes per file:   {mapped_genes_set}")
        print(f"Data type:        {dtypes}")
        print(f"Avg mapping rate: {total_mapped/total_original*100:.1f}%")

        print("\n" + "-"*60)
        print("FILE DETAILS")
        print("-"*60)
        print(f"{'File':<40} {'Cells':>10} {'Original':>10} {'Mapped':>8} {'Rate':>8}")
        print("-"*60)
        for r in success_results:
            rate = r['mapped_genes'] / r['original_genes'] * 100
            print(f"{r['file']:<40} {r['n_cells']:>10,} {r['original_genes']:>10} {r['mapped_genes']:>8} {rate:>7.1f}%")

    if failed_results:
        print("\n" + "-"*60)
        print("FAILED FILES")
        print("-"*60)
        for r in failed_results:
            print(f"  {r['file']}: {r.get('error', 'Unknown error')}")

    print("\n" + "="*60)
    print(f"Output directory: {OUTPUT}")
    print("="*60)


if __name__ == "__main__":
    main()
