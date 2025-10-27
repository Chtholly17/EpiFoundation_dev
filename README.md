# EpiFoundation: Single-Cell Multi-Omics Foundation Model

EpiFoundation is a transformer-based foundation model for single-cell multi-omics data integration, designed to learn representations from paired ATAC-seq and RNA-seq data. This repository contains code for pretraining, finetuning, and evaluation of the model.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Configuration Files](#configuration-files)
- [Training Pipeline](#training-pipeline)
  - [Pretraining](#pretraining)
  - [Finetuning](#finetuning)
  - [Evaluation](#evaluation)
- [Model Architecture](#model-architecture)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Installation

### 1. Create a Python Environment

```bash
# Using conda (recommended)
conda create -n epifoundation python=3.9
conda activate epifoundation

# Or using venv
python -m venv epifoundation
source epifoundation/bin/activate  # On Windows: epifoundation\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install PyTorch

Install PyTorch with CUDA support (adjust CUDA version as needed):

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

### 1. Prepare Your Data

```python
from data.preprocess import Preprocessor
import scanpy as sc

# Load raw data
adata = sc.read_h5ad('path/to/your/data.h5ad')

# Preprocess config
preprocess_config = {
    'path': '/path/to/output/',
    'raw_data': 'your_data.h5ad',
    'use_key': 'X',
    'normalize_total': 1e4,
    'binning': 52,  # Number of bins for expression values
    'result_binned_key': 'X_binned',
    'output_name': 'processed_data'
}

# Run preprocessing
preprocessor = Preprocessor(preprocess_config)
preprocessor.preprocess()
```

### 2. Run Pretraining

```bash
# Single GPU
CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 --master_port 29502 \
    pretrain.py --config ./configs/pretrain/atac_cross_debug.yml

# Multi-GPU (8 GPUs)
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node=8 --master_port 29502 \
    pretrain.py --config ./configs/pretrain/atac_cross_debug.yml
```

### 3. Run Finetuning

```bash
CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 --master_port 29502 \
    finetune.py --config ./configs/finetune/mini_atlas_10bins.yml
```

### 4. Run Evaluation

```bash
CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 --master_port 29502 \
    eval.py --config ./configs/eval/mini_atlas_rna_bmmc.yml
```

## Data Preparation

### Data Format

The model expects data in AnnData (`.h5ad`) format with the following structure:

- **RNA-seq data**: Gene expression matrix with cells as rows and genes as columns
- **ATAC-seq data**: Peak accessibility matrix with cells as rows and peaks as columns
- **Metadata**: Cell type labels, batch information, and other annotations

### Preprocessing Steps

1. **Load raw data**: Start with raw count matrices in `.h5ad` format
2. **Normalization**: Normalize total counts to 10,000 (default)
3. **Binning**: Discretize continuous expression values into bins (e.g., 52 bins)
4. **Vocabulary creation**: Create gene and cell type vocabularies

Example preprocessing workflow:

```python
from data.preprocess import Preprocessor

config = {
    'path': '/path/to/data/',
    'raw_data': 'bmmc_rna.h5ad',
    'use_key': 'X',
    'filter_gene_by_counts': False,
    'filter_cell_by_counts': False,
    'normalize_total': 1e4,
    'result_normed_key': 'X_normed',
    'log1p': False,
    'result_log1p_key': 'X_log1p',
    'subset_hvg': False,
    'binning': 52,
    'result_binned_key': 'X_binned',
    'batch_key': 'orig.ident',
    'output_name': 'bmmc_rna_binned'
}

preprocessor = Preprocessor(config)
preprocessor.preprocess()
```

### Vocabulary Files

You need to create vocabulary files for:
- **RNA genes** (`rna_vocab.json`)
- **ATAC peaks** (`atac_vocab.json`)
- **Cell types** (`cell_type_vocab.json`)
- **Batches** (`batch_vocab.json`)
- **Chromosomes** (`chr_vocab.json`)
- **Gene-to-chromosome mapping** (`gene2chr.json`)

These can be generated using the `prepare_data.py` script.

## Configuration Files

All experiments are configured using YAML files located in the `configs/` directory:

- `configs/pretrain/`: Pretraining configurations
- `configs/finetune/`: Finetuning configurations
- `configs/eval/`: Evaluation configurations

### Configuration Structure

#### Pretraining Configuration Example

```yaml
task_name: my_pretrain_task

train:
  seed: 2002
  batch_size: 8
  lr: 1e-4
  epochs: 150
  gradient_accumulation_steps: 20
  amp: True  # Automatic Mixed Precision
  save_ckpt_freq: 20
  resume: False

  model:
    encoder: transformer
    pretrained: null  # Path to pretrained model or null
    embedding_method: id_only
    atac_max_len: 12000  # Max sequence length for ATAC
    rna_max_len: 8000    # Max sequence length for RNA
    embedding_dim: 512
    num_layers: 6
    head_num: 8
    head_dim: 1024
    dropout: 0.15
    cell_emb_style: cls
    mvc_arch_style: concat query
    use_batch_labels: True
    use_chr_labels: True
  
  task_weight:
    cell_type: 0.0  # Weight for cell type classification loss
    mvc: 1.0        # Weight for masked value prediction loss

valid:
  freq: 2  # Validation frequency (every N epochs)

data:
  bin_num: 2  # Number of bins for expression values
  append_cls: True
  train:
    atac_path: /path/to/train/atac.h5ad
    atac_key: X
    rna_path: /path/to/train/rna.h5ad
    rna_key: X
  test:
    atac_path: /path/to/test/atac.h5ad
    atac_key: X
    rna_path: /path/to/test/rna.h5ad
    rna_key: X

vocab:
  rna_path: /path/to/rna_vocab.json
  atac_path: /path/to/atac_vocab.json
  cell_type_path: /path/to/cell_type_vocab.json
  batch_path: /path/to/batch_vocab.json
  chr_path: /path/to/chr_vocab.json
  gene2chr_path: /path/to/gene2chr.json
  special_tokens:
    pad: {token: <pad>, value: 2}
    mask: {token: <mask>, value: 3}
    cls: {token: <cls>, value: 0}
```

### Key Configuration Parameters

#### Training Parameters

- **batch_size**: Number of samples per batch per GPU
- **lr**: Learning rate (default: 1e-4)
- **epochs**: Number of training epochs
- **gradient_accumulation_steps**: Accumulate gradients over N steps before updating
- **amp**: Use Automatic Mixed Precision (True/False)
- **save_ckpt_freq**: Save checkpoint every N epochs
- **resume**: Resume from last checkpoint (True/False)

#### Model Parameters

- **encoder**: Model architecture ('transformer', 'performer', etc.)
- **pretrained**: Path to pretrained checkpoint (null for training from scratch)
- **embedding_method**: How to embed tokens ('id_only', etc.)
- **atac_max_len**: Maximum sequence length for ATAC data
- **rna_max_len**: Maximum sequence length for RNA data
- **embedding_dim**: Dimension of token embeddings
- **num_layers**: Number of transformer layers
- **head_num**: Number of attention heads
- **head_dim**: Dimension of each attention head
- **dropout**: Dropout probability
- **cell_emb_style**: Cell embedding style ('cls', 'mean', etc.)
- **mvc_arch_style**: Architecture style for masked value prediction
- **use_batch_labels**: Use batch labels for embedding (True/False)
- **use_chr_labels**: Use chromosome labels for embedding (True/False)

#### Task Weights

- **cell_type**: Weight for cell type classification loss
- **mvc**: Weight for masked value prediction loss
- **zero_bce**: Weight for zero-inflated binary cross-entropy loss (if applicable)
- **value_mse**: Weight for continuous value MSE loss (if applicable)

#### Data Parameters

- **bin_num**: Number of bins for discretizing expression values
- **append_cls**: Append [CLS] token to sequences (True/False)
- **atac_path / rna_path**: Paths to preprocessed data files
- **atac_key / rna_key**: Which layer in AnnData to use ('X', 'X_binned', etc.)

#### Vocabulary Parameters

Paths to vocabulary JSON files:
- **rna_path**: RNA gene vocabulary
- **atac_path**: ATAC peak vocabulary
- **cell_type_path**: Cell type vocabulary
- **batch_path**: Batch vocabulary
- **chr_path**: Chromosome vocabulary
- **gene2chr_path**: Gene-to-chromosome mapping
- **hvg_path**: Highly variable genes list (for finetuning)

#### Special Tokens

- **pad**: Padding token and value
- **mask**: Mask token and value (usually bin_num + 1)
- **cls**: Classification token and value

## Training Pipeline

### Pretraining

Pretraining is performed using masked value prediction (MVC) on paired ATAC-seq and RNA-seq data.

#### Command

```bash
# Multi-GPU training
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun \
    --nproc_per_node=8 \
    --master_port 29502 \
    pretrain.py \
    --config ./configs/pretrain/atac_cross_debug.yml
```

#### Key Features

- **Distributed Data Parallel (DDP)**: Efficient multi-GPU training
- **Automatic Mixed Precision (AMP)**: Faster training with lower memory usage
- **Gradient Accumulation**: Train with larger effective batch sizes
- **Cross-modality Learning**: Learn joint representations from ATAC and RNA data

#### Outputs

- **Checkpoints**: Saved to `experiment/<task_name>/ckpts/`
- **Logs**: TensorBoard logs in `experiment/<task_name>/logs/`
- **Pretrained model**: `pretrain.pth` saved at the end of training

### Finetuning

Finetuning adapts the pretrained model to specific downstream tasks like cell type classification or expression prediction.

#### Command

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun \
    --nproc_per_node=8 \
    --master_port 29502 \
    finetune.py \
    --config ./configs/finetune/mini_atlas_10bins.yml
```

#### Finetuning Tasks

1. **Cell Type Classification**: Predict cell type from single-cell profiles
2. **Masked Value Prediction**: Continue pretraining objective
3. **Zero-Inflated Regression**: Predict continuous expression values with zero inflation modeling

#### Configuration for Cell Type Classification

```yaml
task_weight:
  cell_type: 1.0
  mvc: 0.0
```

#### Configuration for Expression Prediction

```yaml
task_weight:
  cell_type: 0.0
  mvc: 1.0
  zero_bce: 1.0  # If using zero-inflated model
  value_mse: 1.0
```

#### Outputs

- **Checkpoints**: Saved to `experiment/<task_name>/ckpts/`
- **Finetuned model**: `finetuned.pth` saved at the end of training
- **WandB logs**: If WandB is configured

### Evaluation

Evaluation measures model performance on held-out test data.

#### Command

```bash
CUDA_VISIBLE_DEVICES="0" torchrun \
    --nproc_per_node=1 \
    --master_port 29502 \
    eval.py \
    --config ./configs/eval/mini_atlas_rna_bmmc.yml
```

#### Evaluation Metrics

For cell type classification:
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Macro and micro F1 scores
- **Confusion Matrix**: Per-class performance

For expression prediction:
- **Pearson Correlation**: Gene-wise correlation
- **Spearman Correlation**: Rank correlation
- **MSE**: Mean squared error
- **R² Score**: Coefficient of determination

#### Outputs

- **Predictions**: Saved to `experiment/<task_name>/predictions.h5ad`
- **Metrics**: JSON file with evaluation metrics
- **Visualizations**: UMAP plots, confusion matrices, etc.

## Model Architecture

EpiFoundation is based on a transformer encoder architecture with the following components:

1. **Token Embeddings**: Embed gene/peak IDs and expression values
2. **Position Embeddings**: Encode sequence position information
3. **Transformer Encoder**: Multi-layer self-attention mechanism
4. **Cross-Modal Attention**: Integrate information across ATAC and RNA modalities
5. **Task-Specific Heads**:
   - Masked value prediction head
   - Cell type classification head
   - Zero-inflated regression head (optional)

### Key Features

- **Multi-omics Integration**: Joint modeling of ATAC and RNA data
- **Flexible Architecture**: Support for different transformer variants
- **Scalability**: Efficient training on large datasets with DDP
- **Transfer Learning**: Pretrain on large datasets, finetune on smaller tasks

## Advanced Usage

### Resuming Training

To resume training from a checkpoint, set `resume: True` in the config file:

```yaml
train:
  resume: True
```

The script will automatically load the latest checkpoint from the experiment directory.

### Custom Learning Rate Schedules

Modify the learning rate schedule in the training script:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
```

### Using Different Transformer Backends

The model supports different attention mechanisms:

- **Flash Attention**: Fast and memory-efficient attention (default)
- **Standard Attention**: PyTorch native attention
- **Performer**: Linear attention approximation

Specify the backend in the config:

```yaml
train:
  model:
    encoder: transformer  # or 'performer'
```

### Monitoring with WandB

To enable Weights & Biases logging, initialize wandb in your script:

```python
import wandb

wandb.init(
    project="epifoundation",
    name=task_name,
    config=config
)
```

### Memory Optimization

For large models or datasets:

1. **Reduce batch size**: Lower `batch_size` in config
2. **Increase gradient accumulation**: Increase `gradient_accumulation_steps`
3. **Use AMP**: Set `amp: True`
4. **Reduce sequence length**: Lower `atac_max_len` and `rna_max_len`
5. **Use gradient checkpointing**: Enable in model config

Example:

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution**: Reduce batch size or sequence length, enable AMP, or use gradient accumulation.

```yaml
train:
  batch_size: 4  # Reduce from 8
  gradient_accumulation_steps: 40  # Increase from 20
  amp: True
```

#### 2. Port Already in Use

**Solution**: Change the `--master_port` argument to a different port number.

```bash
torchrun --master_port 29503 pretrain.py --config ...
```

#### 3. Data Loading Errors

**Solution**: Verify data paths and keys in the config file.

```yaml
data:
  train:
    rna_path: /correct/path/to/rna.h5ad
    rna_key: X  # or X_binned, X_log1p, etc.
```

#### 4. Vocabulary Mismatch

**Solution**: Ensure vocabulary files match the genes/peaks in your data.

#### 5. DDP Initialization Errors

**Solution**: Make sure all GPUs are visible and not occupied by other processes.

```bash
nvidia-smi  # Check GPU availability
```

### Performance Tips

1. **Use multiple GPUs**: Significantly speeds up training
2. **Enable AMP**: Reduces memory usage and increases speed
3. **Optimize data loading**: Use more `num_workers` in DataLoader
4. **Preprocess data once**: Save preprocessed data and reuse
5. **Use flash attention**: Much faster than standard attention

### Getting Help

If you encounter issues:

1. Check the configuration files for typos or incorrect paths
2. Verify data format and preprocessing steps
3. Review the error messages and stack traces
4. Consult the original paper for methodology details

## Citation

If you use this code in your research, please cite:

```bibtex
@article{epifoundation2024,
  title={EpiFoundation: A Foundation Model for Single-Cell Multi-Omics},
  author={Your Name},
  journal={bioRxiv},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

---

**Happy modeling! 🧬🔬**

