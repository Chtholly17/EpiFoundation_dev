#!/bin/bash

# Finetune script example
# This script demonstrates how to run finetuning on pretrained models

# Example 1: Single GPU finetuning
# CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 --master_port 29502 finetune.py --config ./configs/finetune/mini_atlas_10bins.yml

# Example 2: Multi-GPU finetuning (8 GPUs)
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node=8 --master_port 29502 finetune.py --config ./configs/finetune/mini_atlas_10bins.yml

# Example 3: Finetune with specific config
# CUDA_VISIBLE_DEVICES="0,1,2" torchrun --nproc_per_node=3 --master_port 29502 finetune.py --config ./configs/finetune/mini_atlas_zero_inflated.yml

# Parameters explanation:
# - CUDA_VISIBLE_DEVICES: Specify which GPUs to use
# - --nproc_per_node: Number of processes (GPUs) per node
# - --master_port: Port for distributed training communication
# - --config: Path to the configuration YAML file with finetuning settings

