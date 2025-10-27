# Quick Start Guide

Get started with EpiFoundation in 5 minutes!

## Installation (2 minutes)

```bash
# Create environment
conda create -n epifoundation python=3.9
conda activate epifoundation

# Install dependencies
cd EpiFoundation_dev
pip install -r requirements.txt

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Test Your Installation (1 minute)

```bash
# Check if PyTorch can see your GPUs
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

Expected output:
```
CUDA available: True
GPU count: 8
```

## Prepare Your Data (Varies)

### Option 1: Use Example Data

```bash
# Download example data (if available)
wget https://example.com/epifoundation_example_data.tar.gz
tar -xzf epifoundation_example_data.tar.gz
```

### Option 2: Prepare Your Own Data

```python
from data.preprocess import Preprocessor
import scanpy as sc

# Load your data
adata = sc.read_h5ad('your_data.h5ad')

# Quick preprocessing
config = {
    'path': './data/',
    'raw_data': 'your_data.h5ad',
    'use_key': 'X',
    'normalize_total': 1e4,
    'binning': 52,
    'result_binned_key': 'X_binned',
    'output_name': 'processed_data'
}

preprocessor = Preprocessor(config)
preprocessor.preprocess()
```

See [DATA_PREPARATION.md](docs/DATA_PREPARATION.md) for detailed instructions.

## Update Configuration (1 minute)

Edit `configs/pretrain/atac_cross_debug.yml`:

```yaml
# Update these paths to your data
data:
  train:
    atac_path: /path/to/your/train/atac.h5ad
    rna_path: /path/to/your/train/rna.h5ad
  test:
    atac_path: /path/to/your/valid/atac.h5ad
    rna_path: /path/to/your/valid/rna.h5ad

vocab:
  rna_path: /path/to/your/rna_vocab.json
  atac_path: /path/to/your/atac_vocab.json
  cell_type_path: /path/to/your/cell_type_vocab.json
  # ... other vocab paths
```

## Run Training (1 minute to start)

### Pretraining

```bash
# Single GPU (for testing)
CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 --master_port 29502 \
    pretrain.py --config ./configs/pretrain/atac_cross_debug.yml

# Multi-GPU (for production)
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node=8 --master_port 29502 \
    pretrain.py --config ./configs/pretrain/atac_cross_debug.yml
```

Or use the script:

```bash
cd scripts
./pretrain.sh
```

### Finetuning

After pretraining completes, update the config to point to your pretrained model:

```yaml
train:
  model:
    pretrained: /path/to/experiment/<task_name>/ckpts/pretrain.pth
```

Then run:

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node=8 --master_port 29502 \
    finetune.py --config ./configs/finetune/mini_atlas_10bins.yml
```

### Evaluation

```bash
CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 --master_port 29502 \
    eval.py --config ./configs/eval/mini_atlas_rna_bmmc.yml
```

## Monitor Training

### TensorBoard

```bash
tensorboard --logdir experiment/<task_name>/logs/
```

Then open http://localhost:6006 in your browser.

### Weights & Biases (Optional)

Add to your script:

```python
import wandb
wandb.init(project="epifoundation", name="my_experiment")
```

## Check Results

After training, your results will be in:

```
experiment/<task_name>/
├── ckpts/
│   ├── checkpoint_20.pt
│   ├── checkpoint_40.pt
│   └── pretrain.pth  (or finetuned.pth)
├── logs/
│   └── tensorboard_logs
└── config.yml  (copy of your config)
```

## Common Commands Cheat Sheet

```bash
# Check GPU usage
nvidia-smi

# Monitor GPU usage continuously
watch -n 1 nvidia-smi

# Kill a stuck process
pkill -9 python

# Check disk space
df -h

# Check experiment size
du -sh experiment/<task_name>/

# Resume training (set resume: True in config)
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node=8 --master_port 29502 \
    pretrain.py --config ./configs/pretrain/atac_cross_debug.yml

# Change port if busy
torchrun --master_port 29503 pretrain.py --config ...
```

## Next Steps

1. **Read the full documentation**:
   - [README.md](README.md) - Complete guide
   - [DATA_PREPARATION.md](docs/DATA_PREPARATION.md) - Data preprocessing
   - [CONFIGURATION_GUIDE.md](docs/CONFIGURATION_GUIDE.md) - All parameters

2. **Experiment with configurations**:
   - Try different learning rates
   - Adjust batch sizes for your GPU memory
   - Tune task weights for your objective

3. **Monitor and optimize**:
   - Track validation loss
   - Check for overfitting
   - Optimize hyperparameters

4. **Evaluate and analyze**:
   - Run evaluation on test set
   - Visualize embeddings with UMAP
   - Analyze per-gene/per-cell performance

## Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| CUDA OOM | Reduce `batch_size` in config |
| Port in use | Change `--master_port` to different number |
| Slow training | Set `amp: True` in config |
| NaN loss | Lower learning rate (`lr: 1e-5`) |
| Import error | Check virtual environment is activated |
| Config not found | Use absolute paths in config file |

## Getting Help

1. Check error messages carefully
2. Review configuration files
3. Consult documentation
4. Open an issue on GitHub

---

**You're ready to go! Start training your foundation model! 🚀**

