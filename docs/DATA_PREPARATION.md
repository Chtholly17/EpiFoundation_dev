# Data Preparation Guide

This guide walks you through the complete data preparation pipeline for EpiFoundation.

## Overview

The data preparation process involves:
1. Loading raw single-cell data
2. Quality control and filtering
3. Normalization
4. Binning expression values
5. Creating vocabulary files
6. Organizing data for training

## Step-by-Step Guide

### 1. Raw Data Requirements

You need:
- **RNA-seq data**: Single-cell gene expression matrix (`.h5ad` format)
- **ATAC-seq data**: Single-cell chromatin accessibility matrix (`.h5ad` format)
- Both datasets should have matched cells (same cell barcodes)

AnnData structure:
```
adata.X: expression/accessibility matrix
adata.obs: cell metadata (cell types, batches, etc.)
adata.var: gene/peak metadata
```

### 2. Preprocessing RNA-seq Data

```python
import scanpy as sc
from data.preprocess import Preprocessor

# Load raw RNA data
rna_data = sc.read_h5ad('path/to/raw_rna.h5ad')

# Preprocessing configuration
rna_config = {
    'path': '/path/to/output/',
    'raw_data': 'raw_rna.h5ad',
    'use_key': 'X',
    
    # Quality control
    'filter_gene_by_counts': True,
    'min_genes': 200,  # Filter cells with < 200 genes
    'min_cells': 3,    # Filter genes expressed in < 3 cells
    'filter_cell_by_counts': True,
    
    # Normalization
    'normalize_total': 1e4,  # Normalize to 10,000 counts per cell
    'result_normed_key': 'X_normed',
    
    # Log transformation
    'log1p': True,
    'result_log1p_key': 'X_log1p',
    
    # Highly variable genes (optional)
    'subset_hvg': False,
    'hvg_use_key': 'X_log1p',
    'hvg_flavor': 'seurat_v3',
    'n_top_genes': 2000,
    
    # Binning
    'binning': 52,  # Number of bins
    'result_binned_key': 'X_binned',
    
    # Batch key
    'batch_key': 'batch',  # Column in adata.obs
    
    # Output
    'output_name': 'rna_binned'
}

# Run preprocessing
preprocessor = Preprocessor(rna_config)
preprocessor.preprocess()
```

### 3. Preprocessing ATAC-seq Data

```python
# Load raw ATAC data
atac_data = sc.read_h5ad('path/to/raw_atac.h5ad')

# ATAC preprocessing config
atac_config = {
    'path': '/path/to/output/',
    'raw_data': 'raw_atac.h5ad',
    'use_key': 'X',
    
    # For ATAC, typically use binary values
    'normalize_total': None,  # Skip normalization
    'log1p': False,
    
    # Binning (or use binary)
    'binning': 2,  # Binary: 0 (closed), 1 (open)
    'result_binned_key': 'X_binned',
    
    'batch_key': 'batch',
    'output_name': 'atac_binned'
}

preprocessor = Preprocessor(atac_config)
preprocessor.preprocess()
```

### 4. Creating Vocabulary Files

Vocabulary files map genes/peaks/cell types to integer IDs.

#### RNA Gene Vocabulary

```python
import json

# Get unique genes
genes = rna_data.var_names.tolist()

# Create vocabulary
rna_vocab = {gene: idx for idx, gene in enumerate(genes)}

# Save
with open('rna_vocab.json', 'w') as f:
    json.dump(rna_vocab, f)
```

#### ATAC Peak Vocabulary

```python
# Get unique peaks
peaks = atac_data.var_names.tolist()

# Create vocabulary
atac_vocab = {peak: idx for idx, peak in enumerate(peaks)}

# Save
with open('atac_vocab.json', 'w') as f:
    json.dump(atac_vocab, f)
```

#### Cell Type Vocabulary

```python
# Get unique cell types
cell_types = rna_data.obs['cell_type'].unique().tolist()

# Create vocabulary
cell_type_vocab = {ct: idx for idx, ct in enumerate(cell_types)}

# Save
with open('cell_type_vocab.json', 'w') as f:
    json.dump(cell_type_vocab, f)
```

#### Batch Vocabulary

```python
# Get unique batches
batches = rna_data.obs['batch'].unique().tolist()

# Create vocabulary
batch_vocab = {batch: idx for idx, batch in enumerate(batches)}

# Save
with open('batch_vocab.json', 'w') as f:
    json.dump(batch_vocab, f)
```

#### Chromosome Vocabulary

```python
# Get unique chromosomes
chromosomes = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY']

# Create vocabulary
chr_vocab = {chr: idx for idx, chr in enumerate(chromosomes)}

# Save
with open('chr_vocab.json', 'w') as f:
    json.dump(chr_vocab, f)
```

#### Gene-to-Chromosome Mapping

```python
# Assuming gene names contain chromosome info
# Or load from gene annotation file

gene2chr = {}
for gene in genes:
    # Extract chromosome from gene annotation
    # This depends on your data format
    chr = extract_chromosome(gene)  # Implement this function
    gene2chr[gene] = chr

# Save
with open('gene2chr.json', 'w') as f:
    json.dump(gene2chr, f)
```

### 5. Data Splitting

Split data into training and validation sets:

```python
from sklearn.model_selection import train_test_split

# Split cells
train_cells, val_cells = train_test_split(
    rna_data.obs_names,
    test_size=0.1,
    stratify=rna_data.obs['cell_type'],  # Stratified split
    random_state=42
)

# Create train/val datasets
rna_train = rna_data[train_cells].copy()
rna_val = rna_data[val_cells].copy()

atac_train = atac_data[train_cells].copy()
atac_val = atac_data[val_cells].copy()

# Save
rna_train.write_h5ad('train/rna_binned.h5ad')
rna_val.write_h5ad('valid/rna_binned.h5ad')
atac_train.write_h5ad('train/atac_binned.h5ad')
atac_val.write_h5ad('valid/atac_binned.h5ad')
```

### 6. Directory Structure

Organize your data as follows:

```
data/
├── train/
│   ├── rna_binned.h5ad
│   └── atac_binned.h5ad
├── valid/
│   ├── rna_binned.h5ad
│   └── atac_binned.h5ad
└── vocab/
    ├── rna_vocab.json
    ├── atac_vocab.json
    ├── cell_type_vocab.json
    ├── batch_vocab.json
    ├── chr_vocab.json
    └── gene2chr.json
```

### 7. Using prepare_data.py

You can also use the provided `prepare_data.py` script:

```bash
python prepare_data.py \
    --rna_path path/to/raw_rna.h5ad \
    --atac_path path/to/raw_atac.h5ad \
    --output_dir path/to/output/ \
    --binning 52 \
    --test_size 0.1
```

See `prepare_data.py` for more options.

## Binning Strategies

### RNA Expression Binning

Common binning strategies:

1. **Equal-width binning**: Divide expression range into equal bins
   ```python
   bins = np.linspace(0, max_expr, num_bins)
   ```

2. **Equal-frequency binning**: Each bin has approximately the same number of values
   ```python
   bins = np.percentile(expr, np.linspace(0, 100, num_bins))
   ```

3. **Log-scale binning**: Bins are evenly spaced in log scale
   ```python
   bins = np.logspace(0, np.log10(max_expr), num_bins)
   ```

### ATAC Binary Values

For ATAC-seq, typically use binary values:
- 0: Peak closed (no accessibility)
- 1: Peak open (accessible)

Threshold:
```python
atac_binary = (atac_data.X > 0).astype(int)
```

## Quality Control

Before preprocessing, perform quality control:

```python
import scanpy as sc

# Calculate QC metrics
sc.pp.calculate_qc_metrics(adata, inplace=True)

# Filter cells
adata = adata[adata.obs['n_genes_by_counts'] > 200, :]
adata = adata[adata.obs['n_genes_by_counts'] < 5000, :]

# Filter genes
adata = adata[:, adata.var['n_cells_by_counts'] > 3]

# Filter by mitochondrial percentage (for RNA)
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
adata = adata[adata.obs['pct_counts_mt'] < 20, :]
```

## Validation

After preprocessing, validate your data:

```python
# Check data shape
print(f"RNA train: {rna_train.shape}")
print(f"ATAC train: {atac_train.shape}")

# Check binned values
print(f"RNA bins: {np.unique(rna_train.layers['X_binned'])}")
print(f"ATAC bins: {np.unique(atac_train.layers['X_binned'])}")

# Check vocabulary sizes
print(f"RNA vocab size: {len(rna_vocab)}")
print(f"ATAC vocab size: {len(atac_vocab)}")
print(f"Cell type vocab size: {len(cell_type_vocab)}")
```

## Troubleshooting

### Issue: Memory error during preprocessing

**Solution**: Process data in chunks or use sparse matrices.

```python
# Use sparse matrices
import scipy.sparse as sp
adata.X = sp.csr_matrix(adata.X)
```

### Issue: Binning produces unexpected values

**Solution**: Check the binning range and method. Visualize distribution.

```python
import matplotlib.pyplot as plt
plt.hist(adata.X.flatten(), bins=50)
plt.show()
```

### Issue: Vocabulary mismatch

**Solution**: Ensure vocabularies are created from the same gene/peak sets used in training.

## Example: Complete Pipeline

```python
import scanpy as sc
import numpy as np
import json
from sklearn.model_selection import train_test_split

# 1. Load data
rna = sc.read_h5ad('raw_rna.h5ad')
atac = sc.read_h5ad('raw_atac.h5ad')

# 2. QC
sc.pp.filter_cells(rna, min_genes=200)
sc.pp.filter_genes(rna, min_cells=3)

# 3. Normalize RNA
sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)

# 4. Bin RNA
from data.preprocess import bin_expression
rna.layers['X_binned'] = bin_expression(rna.X, n_bins=52)

# 5. Binarize ATAC
atac.layers['X_binned'] = (atac.X > 0).astype(int)

# 6. Split data
train_idx, val_idx = train_test_split(
    np.arange(rna.n_obs),
    test_size=0.1,
    stratify=rna.obs['cell_type'],
    random_state=42
)

# 7. Save
rna[train_idx].write_h5ad('train/rna.h5ad')
rna[val_idx].write_h5ad('valid/rna.h5ad')
atac[train_idx].write_h5ad('train/atac.h5ad')
atac[val_idx].write_h5ad('valid/atac.h5ad')

# 8. Create vocabularies
rna_vocab = {g: i for i, g in enumerate(rna.var_names)}
atac_vocab = {p: i for i, p in enumerate(atac.var_names)}
cell_vocab = {c: i for i, c in enumerate(rna.obs['cell_type'].unique())}

with open('vocab/rna_vocab.json', 'w') as f:
    json.dump(rna_vocab, f)
with open('vocab/atac_vocab.json', 'w') as f:
    json.dump(atac_vocab, f)
with open('vocab/cell_type_vocab.json', 'w') as f:
    json.dump(cell_vocab, f)

print("Data preparation complete!")
```

## Next Steps

After preparing your data:

1. Update the configuration file with your data paths
2. Verify all paths and vocabulary files
3. Start pretraining with `pretrain.py`

See the main README for training instructions.

