import os
import re
import logging
import numpy as np
import pandas as pd
import anndata
from pybiomart import Dataset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def load_ensembl_annotations():
    human_ds = Dataset(name="hsapiens_gene_ensembl", host="http://www.ensembl.org")
    hdf = human_ds.query(attributes=["ensembl_gene_id", "external_gene_name"]).dropna()
    ens2sym = dict(zip(hdf["Gene stable ID"], hdf["Gene name"]))
    valid_hs = set(hdf["Gene name"].str.upper())
    return ens2sym, valid_hs


clean_gene = lambda g: re.sub(r"\.\d+$", "", g).upper()
is_ensembl_id = lambda s: bool(re.fullmatch(r"ENSG\d{11}", s))


def map_gene(g, ens2sym, valid_hs):
    g2 = clean_gene(g)
    if is_ensembl_id(g2):
        return ens2sym.get(g2)
    if g2 in valid_hs:
        return g2
    return None


def select_group_shared_hvg(h5ad_list, st_path, ens2sym, valid_hs, HVG_K):
    gene_exprs = []
    shared_genes = None

    for slide_id in h5ad_list:
        path = os.path.join(st_path, f"{slide_id}.h5ad")
        if not os.path.exists(path):
            logging.warning(f"{slide_id} not found, skipped.")
            continue

        adata = anndata.read_h5ad(path)
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        genes = np.array(adata.var_names)
        mapped = np.array([map_gene(g, ens2sym, valid_hs) for g in genes])
        mask = pd.notnull(mapped)
        mapped_genes = mapped[mask]
        X = X[:, mask]

        # Log1p only (no library size normalization)
        X_log = np.log1p(X)

        df = pd.DataFrame(X_log, columns=mapped_genes)
        df = df.loc[:, ~df.columns.duplicated()]
        gene_exprs.append(df)

        if shared_genes is None:
            shared_genes = set(df.columns)
        else:
            shared_genes &= set(df.columns)

    if not gene_exprs or not shared_genes:
        logging.warning("No valid shared genes found.")
        return []

    shared_genes = sorted(shared_genes)
    merged_expr = pd.concat([df[shared_genes] for df in gene_exprs], axis=0)
    gene_vars = merged_expr.values.var(axis=0)
    topk_idx = np.argsort(gene_vars)[::-1][:HVG_K]
    return [shared_genes[i] for i in topk_idx]


def save_shared_hvg_expression(h5ad_list, st_path, out_dir, ens2sym, valid_hs, hvg_genes):
    os.makedirs(out_dir, exist_ok=True)
    for slide_id in h5ad_list:
        path = os.path.join(st_path, f"{slide_id}.h5ad")
        if not os.path.exists(path):
            logging.warning(f"{slide_id} not found, skipped.")
            continue

        adata = anndata.read_h5ad(path)
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        genes = np.array(adata.var_names)
        mapped = np.array([map_gene(g, ens2sym, valid_hs) for g in genes])
        mask = pd.notnull(mapped)
        mapped_genes = mapped[mask]
        X = X[:, mask]

        # Log1p only (no library size normalization)
        X_log = np.log1p(X)

        df = pd.DataFrame(X_log, columns=mapped_genes)
        df = df.loc[:, ~df.columns.duplicated()]

        if not all(g in df.columns for g in hvg_genes):
            logging.warning(f"{slide_id} missing some HVG genes. Skipped.")
            continue

        selected_expr = df[hvg_genes].values
        out_data = {
            "gene_names": np.array(hvg_genes),
            "expression": selected_expr.astype(np.float32)
        }

        out_file = os.path.join(out_dir, f"{slide_id}.npy")
        np.save(out_file, out_data)
        logging.info(f"[Saved] {slide_id} -> {out_file}")


def run_all():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ST_DATA = os.environ.get("ST_DATA_PATH", "./raw_data/st")
    HVG_K = 1000
    base_out_dir = os.path.join(SCRIPT_DIR, "hvg_expr_npy")

    ens2sym, valid_hs = load_ensembl_annotations()

    def make_range(prefix, start, end):
        if start <= end:
            return [f"{prefix}{i}" for i in range(start, end + 1)]
        else:
            return [f"{prefix}{i}" for i in range(start, end - 1, -1)]

    tasks = [
        ("HER2",   make_range("SPA", 154, 119)),
        ("BC",     make_range("SPA", 118, 51)),
        ("Kidney", make_range("NCBI", 692, 714))
    ]

    for group_name, h5ad_list in tasks:
        out_dir = os.path.join(base_out_dir, group_name)
        logging.info(f"Processing group: {group_name} -> {len(h5ad_list)} files")

        hvg_genes = select_group_shared_hvg(h5ad_list, ST_DATA, ens2sym, valid_hs, HVG_K)
        if not hvg_genes:
            logging.warning(f"No HVGs selected for group {group_name}, skipping.")
            continue

        save_shared_hvg_expression(h5ad_list, ST_DATA, out_dir, ens2sym, valid_hs, hvg_genes)

    logging.info("All groups finished.")


if __name__ == "__main__":
    run_all()
