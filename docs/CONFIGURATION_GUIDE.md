# Configuration Guide

This guide provides detailed explanations of all configuration parameters used in EpiFoundation.

## Configuration File Structure

All experiments are controlled by YAML configuration files in the `configs/` directory:

- `configs/pretrain/`: Pretraining configurations
- `configs/finetune/`: Finetuning configurations
- `configs/eval/`: Evaluation configurations

## Complete Configuration Reference

### Task Name

```yaml
task_name: my_experiment_name
```

- **Description**: Name of the experiment
- **Type**: String
- **Usage**: Creates a directory `experiment/<task_name>/` for outputs
- **Example**: `mini_atlas_pretrain`, `bmmc_finetune`

---

## Training Configuration

### Basic Training Parameters

```yaml
train:
  local_rank: 0
  seed: 2002
  batch_size: 8
  lr: 1e-4
  epochs: 150
  gradient_accumulation_steps: 20
  amp: True
  save_ckpt_freq: 20
  resume: False
```

#### local_rank
- **Description**: Local rank for distributed training (automatically set by torchrun)
- **Type**: Integer
- **Default**: 0
- **Note**: Don't modify this manually

#### seed
- **Description**: Random seed for reproducibility
- **Type**: Integer
- **Default**: 2002
- **Range**: Any positive integer
- **Note**: Set the same seed for reproducible results

#### batch_size
- **Description**: Number of samples per batch per GPU
- **Type**: Integer
- **Default**: 8
- **Typical Range**: 4-32
- **Note**: Effective batch size = batch_size × num_gpus × gradient_accumulation_steps
- **Memory Impact**: Higher = more GPU memory usage

#### lr (Learning Rate)
- **Description**: Learning rate for optimizer
- **Type**: Float
- **Default**: 1e-4
- **Typical Range**: 1e-5 to 1e-3
- **Recommendations**:
  - Pretraining: 1e-4
  - Finetuning: 1e-5 to 5e-5
- **Note**: Use lower LR for finetuning than pretraining

#### epochs
- **Description**: Number of training epochs
- **Type**: Integer
- **Default**: 150
- **Typical Range**: 50-300
- **Note**: Monitor validation loss to determine optimal number

#### gradient_accumulation_steps
- **Description**: Number of steps to accumulate gradients before updating
- **Type**: Integer
- **Default**: 20
- **Usage**: Simulates larger batch sizes without increasing memory
- **Effective Batch Size**: batch_size × gradient_accumulation_steps × num_gpus
- **Example**: 8 × 20 × 8 = 1280 effective batch size

#### amp (Automatic Mixed Precision)
- **Description**: Use mixed precision training (FP16/FP32)
- **Type**: Boolean
- **Default**: True
- **Benefits**:
  - Faster training (2-3x speedup)
  - Lower memory usage (~50% reduction)
  - Minimal accuracy impact
- **Requirements**: CUDA compute capability >= 7.0

#### save_ckpt_freq
- **Description**: Save checkpoint every N epochs
- **Type**: Integer
- **Default**: 20
- **Note**: Lower values save more frequently but use more disk space

#### resume
- **Description**: Resume training from last checkpoint
- **Type**: Boolean
- **Default**: False
- **Usage**: Set to True to continue interrupted training

---

## Model Configuration

```yaml
train:
  model:
    encoder: transformer
    pretrained: /path/to/pretrained.pth
    embedding_method: id_only
    atac_max_len: 12000
    rna_max_len: 8000
    embedding_dim: 512
    num_layers: 6
    head_num: 8
    head_dim: 1024
    dropout: 0.15
    cell_emb_style: cls
    mvc_arch_style: concat query
    use_batch_labels: True
    use_chr_labels: True
```

#### encoder
- **Description**: Transformer architecture type
- **Type**: String
- **Options**:
  - `transformer`: Standard transformer with flash attention
  - `performer`: Performer with linear attention
- **Default**: `transformer`
- **Recommendation**: Use `transformer` for best performance

#### pretrained
- **Description**: Path to pretrained model checkpoint
- **Type**: String or null
- **Usage**:
  - Pretraining: Set to `null`
  - Finetuning: Set to path of pretrained model
- **Example**: `/path/to/experiment/pretrain/ckpts/pretrain.pth`

#### embedding_method
- **Description**: Method for embedding gene/peak IDs
- **Type**: String
- **Options**: `id_only`
- **Default**: `id_only`
- **Note**: Currently only `id_only` is fully supported

#### atac_max_len
- **Description**: Maximum sequence length for ATAC data
- **Type**: Integer
- **Default**: 12000
- **Typical Range**: 10000-20000
- **Memory Impact**: Linear memory growth with sequence length
- **Note**: Should be >= longest ATAC sequence in your data

#### rna_max_len
- **Description**: Maximum sequence length for RNA data
- **Type**: Integer
- **Default**: 8000
- **Typical Range**: 5000-10000
- **Note**: Should be >= longest RNA sequence in your data

#### embedding_dim
- **Description**: Dimension of token embeddings
- **Type**: Integer
- **Default**: 512
- **Typical Range**: 256-768
- **Trade-offs**:
  - Higher = more capacity, more memory
  - Lower = faster, less capacity
- **Recommendation**: 512 for most tasks

#### num_layers
- **Description**: Number of transformer layers
- **Type**: Integer
- **Default**: 6
- **Typical Range**: 4-12
- **Trade-offs**:
  - More layers = deeper model, better performance, slower
  - Fewer layers = faster, less capacity
- **Recommendation**: 6-8 for most tasks

#### head_num
- **Description**: Number of attention heads
- **Type**: Integer
- **Default**: 8
- **Typical Range**: 4-16
- **Constraint**: embedding_dim must be divisible by head_num
- **Recommendation**: 8 for most tasks

#### head_dim
- **Description**: Dimension of each attention head's query/key/value
- **Type**: Integer
- **Default**: 1024
- **Typical Range**: 512-2048
- **Note**: This is separate from embedding_dim

#### dropout
- **Description**: Dropout probability for regularization
- **Type**: Float
- **Default**: 0.15
- **Typical Range**: 0.1-0.3
- **Usage**:
  - Higher dropout for smaller datasets
  - Lower dropout for larger datasets
- **Recommendation**: 0.15 for pretraining, 0.1-0.2 for finetuning

#### cell_emb_style
- **Description**: Style for generating cell embeddings
- **Type**: String
- **Options**:
  - `cls`: Use [CLS] token embedding
  - `mean`: Mean pooling over all tokens
- **Default**: `cls`
- **Recommendation**: `cls` works well for most tasks

#### mvc_arch_style
- **Description**: Architecture style for masked value prediction
- **Type**: String
- **Options**:
  - `concat query`: Concatenate embeddings before prediction
  - `gene-specific concat query`: Gene-specific decoders
- **Default**: `concat query`
- **Note**: Use `gene-specific` for better expression prediction

#### use_batch_labels
- **Description**: Include batch information in embeddings
- **Type**: Boolean
- **Default**: True
- **Usage**:
  - True: Model learns to correct batch effects
  - False: Ignore batch information
- **Recommendation**: True for multi-batch data

#### use_chr_labels
- **Description**: Include chromosome information in embeddings
- **Type**: Boolean
- **Default**: True
- **Usage**: Helps model learn genomic structure
- **Recommendation**: True for RNA data with gene positions

---

## Task Weights

```yaml
train:
  task_weight:
    cell_type: 1.0
    mvc: 1.0
    zero_bce: 1.0
    value_mse: 1.0
```

#### cell_type
- **Description**: Weight for cell type classification loss
- **Type**: Float
- **Range**: 0.0-10.0
- **Usage**:
  - 0.0: Disable cell type classification
  - > 0.0: Enable with specified weight
- **Typical Values**:
  - Pretraining: 0.0 (unsupervised)
  - Cell type finetuning: 1.0-2.0

#### mvc (Masked Value Prediction)
- **Description**: Weight for masked value prediction loss
- **Type**: Float
- **Range**: 0.0-10.0
- **Usage**:
  - Primary pretraining objective
  - Can be used in finetuning for regularization
- **Typical Values**:
  - Pretraining: 1.0
  - Finetuning (task-specific): 0.0-0.5

#### zero_bce
- **Description**: Weight for zero-inflated binary classification loss
- **Type**: Float
- **Range**: 0.0-10.0
- **Usage**: For zero-inflated expression modeling
- **Note**: Only used with zero-inflated models

#### value_mse
- **Description**: Weight for continuous value MSE loss
- **Type**: Float
- **Range**: 0.0-10.0
- **Usage**: For regression tasks
- **Note**: Only used with zero-inflated models

---

## Validation Configuration

```yaml
valid:
  freq: 2
```

#### freq
- **Description**: Validation frequency (every N epochs)
- **Type**: Integer
- **Default**: 2
- **Usage**: Lower = more frequent validation, slower training
- **Recommendation**: 2-5 for most tasks

---

## Data Configuration

```yaml
data:
  bin_num: 2
  append_cls: True
  train:
    atac_path: /path/to/train/atac.h5ad
    atac_key: X
    rna_path: /path/to/train/rna.h5ad
    rna_key: X
  test:
    atac_path: /path/to/valid/atac.h5ad
    atac_key: X
    rna_path: /path/to/valid/rna.h5ad
    rna_key: X
```

#### bin_num
- **Description**: Number of bins used in preprocessing
- **Type**: Integer
- **Default**: 2
- **Common Values**:
  - 2: Binary (ATAC)
  - 10-52: RNA expression
- **Note**: Must match the binning used in preprocessing
- **Important**: Special tokens use values beyond bin_num

#### append_cls
- **Description**: Append [CLS] token to sequences
- **Type**: Boolean
- **Default**: True
- **Usage**: Required for cell-level predictions
- **Recommendation**: Always True

#### atac_path / rna_path
- **Description**: Path to preprocessed data files
- **Type**: String (absolute or relative path)
- **Format**: `.h5ad` files
- **Example**: `/data/ours/train/atac_binned.h5ad`

#### atac_key / rna_key
- **Description**: Which layer in AnnData to use
- **Type**: String
- **Options**:
  - `X`: Main data matrix
  - `X_binned`: Binned data
  - `X_log1p`: Log-transformed data
- **Default**: `X`
- **Recommendation**: Use layer with binned data

---

## Vocabulary Configuration

```yaml
vocab:
  rna_path: /path/to/rna_vocab.json
  atac_path: /path/to/atac_vocab.json
  cell_type_path: /path/to/cell_type_vocab.json
  batch_path: /path/to/batch_vocab.json
  chr_path: /path/to/chr_vocab.json
  gene2chr_path: /path/to/gene2chr.json
  hvg_path: /path/to/hvg.csv
  special_tokens:
    pad: {token: <pad>, value: 10}
    mask: {token: <mask>, value: 11}
    cls: {token: <cls>, value: 0}
```

#### rna_path
- **Description**: RNA gene vocabulary (gene → ID mapping)
- **Type**: String (path to JSON file)
- **Format**: `{"gene1": 0, "gene2": 1, ...}`

#### atac_path
- **Description**: ATAC peak vocabulary (peak → ID mapping)
- **Type**: String (path to JSON file)
- **Format**: `{"chr1:1000-2000": 0, ...}`

#### cell_type_path
- **Description**: Cell type vocabulary (cell type → ID mapping)
- **Type**: String (path to JSON file)
- **Format**: `{"T cell": 0, "B cell": 1, ...}`

#### batch_path
- **Description**: Batch vocabulary (batch → ID mapping)
- **Type**: String (path to JSON file)
- **Format**: `{"batch1": 0, "batch2": 1, ...}`

#### chr_path
- **Description**: Chromosome vocabulary (chr → ID mapping)
- **Type**: String (path to JSON file)
- **Format**: `{"chr1": 0, "chr2": 1, ...}`

#### gene2chr_path
- **Description**: Gene-to-chromosome mapping
- **Type**: String (path to JSON file)
- **Format**: `{"gene1": "chr1", "gene2": "chr2", ...}`

#### hvg_path (optional)
- **Description**: List of highly variable genes
- **Type**: String (path to CSV file)
- **Usage**: For finetuning on subset of genes
- **Format**: CSV with gene names

#### special_tokens

##### pad (Padding Token)
- **token**: Symbol for padding token (e.g., `<pad>`)
- **value**: Integer value for padding
- **Usage**: Pad sequences to same length
- **Typical Value**: bin_num or higher

##### mask (Mask Token)
- **token**: Symbol for mask token (e.g., `<mask>`)
- **value**: Integer value for masking
- **Usage**: Mask tokens during training
- **Typical Value**: bin_num + 1

##### cls (Classification Token)
- **token**: Symbol for CLS token (e.g., `<cls>`)
- **value**: Integer value for CLS token
- **Usage**: Cell-level representation
- **Typical Value**: 0

**Important**: Special token values must not overlap with binned data values (0 to bin_num-1)

---

## Example Configurations

### Pretraining Configuration

```yaml
task_name: pretrain_pbmc

train:
  seed: 2002
  batch_size: 8
  lr: 1e-4
  epochs: 150
  gradient_accumulation_steps: 20
  amp: True
  save_ckpt_freq: 20
  resume: False

  model:
    encoder: transformer
    pretrained: null
    embedding_dim: 512
    num_layers: 6
    head_num: 8
    head_dim: 1024
    dropout: 0.15
    atac_max_len: 12000
    rna_max_len: 8000
    use_batch_labels: True
    use_chr_labels: True
  
  task_weight:
    cell_type: 0.0
    mvc: 1.0

valid:
  freq: 2

data:
  bin_num: 2
  append_cls: True
  train:
    atac_path: /data/train/atac.h5ad
    atac_key: X
    rna_path: /data/train/rna.h5ad
    rna_key: X
  test:
    atac_path: /data/valid/atac.h5ad
    atac_key: X
    rna_path: /data/valid/rna.h5ad
    rna_key: X

vocab:
  rna_path: /data/vocab/rna_vocab.json
  atac_path: /data/vocab/atac_vocab.json
  cell_type_path: /data/vocab/cell_type_vocab.json
  batch_path: /data/vocab/batch_vocab.json
  chr_path: /data/vocab/chr_vocab.json
  gene2chr_path: /data/vocab/gene2chr.json
  special_tokens:
    pad: {token: <pad>, value: 2}
    mask: {token: <mask>, value: 3}
    cls: {token: <cls>, value: 0}
```

### Finetuning Configuration (Cell Type Classification)

```yaml
task_name: finetune_celltype

train:
  seed: 2002
  batch_size: 8
  lr: 5e-5  # Lower LR for finetuning
  epochs: 50  # Fewer epochs
  gradient_accumulation_steps: 20
  amp: True
  save_ckpt_freq: 10
  resume: False

  model:
    encoder: transformer
    pretrained: /path/to/pretrain.pth  # Load pretrained model
    embedding_dim: 512
    num_layers: 6
    head_num: 8
    head_dim: 1024
    dropout: 0.1  # Lower dropout
    atac_max_len: 12000
    rna_max_len: 8000
    use_batch_labels: False
    use_chr_labels: True
  
  task_weight:
    cell_type: 1.0  # Focus on cell type
    mvc: 0.0

# ... rest similar to pretraining
```

### Evaluation Configuration

```yaml
task_name: eval_expression

train:
  seed: 2002
  batch_size: 16  # Can use larger batch for evaluation
  
  model:
    encoder: transformer
    pretrained: /path/to/finetuned.pth
    # ... other params same as training

# ... data and vocab same as training
```

---

## Tips and Best Practices

### Memory Optimization

1. **Reduce batch size**: Lower `batch_size`
2. **Increase gradient accumulation**: Higher `gradient_accumulation_steps`
3. **Enable AMP**: Set `amp: True`
4. **Reduce sequence length**: Lower `atac_max_len` and `rna_max_len`
5. **Fewer layers**: Lower `num_layers`

### Hyperparameter Tuning

1. **Learning rate**: Most important hyperparameter
   - Start with 1e-4 for pretraining
   - Use 1e-5 to 5e-5 for finetuning
   - Tune using validation loss

2. **Batch size**: Affects training stability
   - Effective batch size should be 512-2048
   - Use gradient accumulation for larger effective batch sizes

3. **Dropout**: Regularization strength
   - Increase for smaller datasets
   - Decrease for larger datasets

4. **Task weights**: Balance multiple objectives
   - Start with equal weights (1.0)
   - Adjust based on validation performance

### Reproducibility

1. Set `seed` to same value
2. Use same number of GPUs
3. Same data preprocessing
4. Same configuration file

### Monitoring Training

1. Watch validation loss (should decrease)
2. Check for overfitting (train loss << valid loss)
3. Monitor GPU utilization (should be >80%)
4. Track memory usage

### Common Issues

1. **NaN loss**: Lower learning rate or enable gradient clipping
2. **OOM error**: Reduce batch size or sequence length
3. **Slow training**: Ensure AMP is enabled, check GPU utilization
4. **Poor performance**: Check data preprocessing, try different hyperparameters

---

This completes the configuration guide. See README.md for usage examples.

