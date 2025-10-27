#!/bin/bash

# Evaluation script example
# This script demonstrates how to run evaluation on trained models

# Example 1: Evaluate cell type classification
# CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 --master_port 29502 eval.py --config ./configs/eval/mini_atlas_celltype_eval.yml

# Example 2: Evaluate expression prediction
CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 --master_port 29502 eval.py --config ./configs/eval/mini_atlas_rna_bmmc.yml

# Example 3: Multi-GPU evaluation
# CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 --master_port 29502 eval.py --config ./configs/eval/mini_atlas_pbmc_eval.yml

# Parameters explanation:
# - CUDA_VISIBLE_DEVICES: Specify which GPUs to use
# - --nproc_per_node: Number of processes (GPUs) per node
# - --master_port: Port for distributed training communication
# - --config: Path to the evaluation configuration YAML file
# - --backend: (Optional) Fast Transformer backend, default is 'flash'

