#!/bin/bash

# Pretrain script example
# This script demonstrates how to run pretraining with distributed data parallel (DDP)

# Set memory allocation configuration
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Example 1: Single GPU training
# CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 --master_port 29502 pretrain.py --config ./configs/pretrain/atac_cross_debug.yml

# Example 2: Multi-GPU training (8 GPUs)
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node=8 --master_port 29502 pretrain.py --config ./configs/pretrain/atac_cross_debug.yml

# Parameters explanation:
# - CUDA_VISIBLE_DEVICES: Specify which GPUs to use (e.g., "0,1,2,3")
# - --nproc_per_node: Number of processes (GPUs) per node
# - --master_port: Port for distributed training communication (change if port is busy)
# - --config: Path to the configuration YAML file

