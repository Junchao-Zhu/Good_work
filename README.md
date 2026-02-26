# DUET: Dual-Paradigm Adaptive Expert Triage with Cellular Inductive Prior for Accurate Spatial Transcriptomics Prediction

## Repository Structure

```
.
├── requirements.txt         # Python dependencies
├── config.py                # Model and dataset configuration
├── train.py                 
├── run_train.sh             
├── data_loader.py           # Data loading and augmentation
├── eval_metric.py           
├── cross_fold.py            
├── models/
│   ├── backbones.py         # DenseNet-121 and ViT-B (CONCH) backbones
│   ├── heads.py             
│   └── model.py             # DualBackboneModel
├── preprocess/
│   ├── library_norm_sc_data_fast.py   
│   ├── map_genes.py                   # Ensembl ID to Gene Symbol mapping
│   ├── compute_gene_overlap.py        
│   ├── filter_hvg_by_overlap.py       
│   ├── filter_sc_by_overlap.py        
│   ├── merge_expression.py            # Merge spatial data with HVG expression
│   └── get_validation_data.py      
└── cell_deconv/
    ├── train_sc.py                    # Cell2location SC signature training
    ├── prepare_st_for_deconv.py       # ST data preparation for deconvolution
    ├── prepare_hvg_subset.py         
    ├── batch_inference.py            
    └── run_deconv.sh                 
```

## Quick Start

### 1. Data Preprocessing

```bash
cd preprocess

# Normalize single-cell data
python library_norm_sc_data_fast.py

# Map gene names (Ensembl ID -> Gene Symbol)
python map_genes.py

# Compute ST-SC gene overlap and filter
python compute_gene_overlap.py
python filter_hvg_by_overlap.py
python filter_sc_by_overlap.py

# Merge spatial format with HVG expression
python merge_expression.py

# Extract validation data
python get_validation_data.py
```

### 2. Cell Type Deconvolution

```bash
cd cell_deconv

# Train cell2location signature model
CUDA_VISIBLE_DEVICES=0 python train_sc.py --tissue breast
CUDA_VISIBLE_DEVICES=1 python train_sc.py --tissue kidney

# Prepare ST data for deconvolution
python prepare_st_for_deconv.py --hvg_k 2000

# (Optional) Select HVG subset
python prepare_hvg_subset.py --hvg_k 150

# Run batch deconvolution (multi-GPU)
bash run_deconv.sh
```

### 3. Model Training

```bash
# Single run
CUDA_VISIBLE_DEVICES=0 python train.py --dataset HER2 --mask_ratio 100 --gpu 0

# Full training (3 datasets x 3 mask ratios, 3 GPUs)
bash run_train.sh
```

**Arguments:**

| Argument | Options | Description |
|---|---|---|
| `--dataset` | `HER2`, `BC`, `Kidney` | Target dataset |
| `--mask_ratio` | `100`, `300`, `500` | Number of target genes to predict |
| `--gpu` | `0`, `1`, `2`, ... | GPU device ID |

## Requirements

Python 3.8+ is required. Install dependencies:

```bash
pip install -r requirements.txt
```
