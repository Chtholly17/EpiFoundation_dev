# -*- coding: utf-8 -*-

import os
import gc
import argparse
import json
import random
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from model import scCross, scTransformer
from loss.loss import MaskedMSELoss
from data import PairedSCDataset
from tokenizer import GeneVocab
import scanpy as sc
import anndata as ad
from utils import *
from memory_profiler import profile

import yaml
import wandb

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default='./configs/pretrain/atac_cross_debug.yml', help='Config file.')
args = parser.parse_args()

# @profile(precision=4, stream=open("memory_profiler.log", "w+"))
def main():
    # read and parse config file
    local_rank = int(os.environ["LOCAL_RANK"])
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    
    train_config = config['train']
    valid_config = config['valid']
    data_config = config['data']
    vocab_config = config['vocab']
    task_name = config['task_name']
    task_floder = './experiment/{}'.format(task_name)
    ckpt_dir = os.path.join(task_floder, 'ckpts')
    

    
    random_seed = train_config['seed']
    EPOCHS = train_config['epochs']
    BATCH_SIZE = train_config['batch_size']
    GRADIENT_ACCUMULATION = train_config['gradient_accumulation_steps']
    LEARNING_RATE = float(train_config['lr'])

    model_name = train_config['model']['encoder']
    
    save_ckpt_freq = train_config['save_ckpt_freq'] if 'save_ckpt_freq' in train_config else 5
    resume = train_config['resume'] if 'resume' in train_config else False
    
    # special tokens
    pad = vocab_config['special_tokens']['pad']
    mask = vocab_config['special_tokens']['mask']
    cls = vocab_config['special_tokens']['cls']
    
    # distibuted setting
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()
    seed_all(random_seed + torch.distributed.get_rank())
    is_master = (local_rank == 0)
    
    # init loggers
    logger = set_log(log_dir= os.path.join(task_floder, 'logs'))
    tb_logger = SummaryWriter(os.path.join(task_floder, 'tb_logs'))
    
    # init wandb (only on master process)
    if is_master:
        wandb.init(
            project="scMultiomics_zero_inflated",
            name=task_name,
            config=config,
            dir=task_floder,
            resume="allow"
        )
        logger.info(dict2str(config))
    
    
    rna_vocab = GeneVocab.from_file(vocab_config['rna_path'])
    atac_vocab = GeneVocab.from_file(vocab_config['atac_path'])
    cell_vocab = GeneVocab.from_file(vocab_config['cell_type_path'])
    batch_vocab = GeneVocab.from_file(vocab_config['batch_path'])
    chr_vocab = GeneVocab.from_file(vocab_config['chr_path'])
    
    if vocab_config['hvg_path'] is not None:
        # read the hvg csv file
        hvg_df = pd.read_csv(vocab_config['hvg_path'])
        # get the first column as the gene names
        hvg_genes = hvg_df.iloc[:, 0].tolist()
    else:
        hvg_genes = None
    
    if is_master:
        logger.info(f'Rna vocab size: {len(rna_vocab)}')
        logger.info(f'Atac vocab size: {len(atac_vocab)}')
        
    if is_master:
        logger.info('loading training data')
    
    train_set = PairedSCDataset(
        rna_file = data_config['train']['rna_path'],
        atac_file= data_config['train']['atac_path'],
        rna_key = data_config['train']['rna_key'],
        atac_key = data_config['train']['atac_key'],
        rna_vocab = rna_vocab,
        atac_vocab = atac_vocab,
        cell_vocab = cell_vocab,
        batch_vocab= batch_vocab,
        chr_vocab = chr_vocab,
        gene2chr_file= vocab_config['gene2chr_path'],
        rna_max_len = train_config['model']['rna_max_len'],
        atac_max_len = train_config['model']['atac_max_len'],
        pad_token = pad['token'],
        rna_pad_value = pad['value'],
        cls_token = cls['token'],
        logger = logger,
        hvg_list=hvg_genes
    )
                                
    gc.collect()
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler, prefetch_factor=4, num_workers=4)
    # train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler)
    
    if is_master:
        logger.info('loading validation data')
    val_set = PairedSCDataset(
        rna_file = data_config['test']['rna_path'],
        atac_file= data_config['test']['atac_path'],
        rna_key = data_config['test']['rna_key'],
        atac_key = data_config['test']['atac_key'],
        rna_vocab = rna_vocab,
        atac_vocab = atac_vocab,
        cell_vocab = cell_vocab,
        batch_vocab= batch_vocab,
        chr_vocab = chr_vocab,
        gene2chr_file= vocab_config['gene2chr_path'],
        rna_max_len = train_config['model']['rna_max_len'],
        atac_max_len = train_config['model']['atac_max_len'],
        pad_token = pad['token'],
        rna_pad_value = pad['value'],
        cls_token = cls['token'],
        logger = logger,
        hvg_list=hvg_genes
    )
    gc.collect()

    val_sampler = SequentialDistributedSampler(val_set, batch_size=BATCH_SIZE, world_size=world_size)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, sampler=val_sampler, prefetch_factor=4, num_workers=4)
    # val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, sampler=val_sampler)
    
    if is_master:
        logger.info('Creating model')
    
    model = scCross(
        num_class_cell = len(cell_vocab),
        num_rnas = len(rna_vocab),
        num_atacs = len(atac_vocab),
        num_values= data_config['bin_num'],
        num_chrs= len(chr_vocab),
        embed_dim = train_config['model']['embedding_dim'],
        depth = train_config['model']['num_layers'],
        heads = train_config['model']['head_num'],
        head_dim = train_config['model']['head_dim'],
        encoder = model_name,
        dropout = train_config['model']['dropout'],
        pad_token_idx_rna = rna_vocab[pad['token']],
        pad_token_idx_atac = atac_vocab[pad['token']],
        cell_emb_style = train_config['model']['cell_emb_style'],
        mvc_arch_style = train_config['model']['mvc_arch_style'],
        use_batch_labels = train_config['model']['use_batch_labels'],
        batch_label_num= len(batch_vocab),
        use_chr_labels= train_config['model']['use_chr_labels'],
        stage= 'value_finetune',
        gene_specific_fp16= train_config['model'].get('gene_specific_fp16', False),
        use_zero_inflated= train_config['model'].get('use_zero_inflated', False),
        regression_max_value= train_config['model'].get('regression_max_value', 10.0),
    ).to(device)
    
    # optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    # learning rate scheduler
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=15,
        cycle_mult=2,
        max_lr=LEARNING_RATE,
        min_lr=1e-6,
        warmup_steps=5,
        gamma=0.9
    )
    
    start_epoch = 1
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Watch model with wandb (only on master)
    if is_master:
        wandb.watch(model, log='all', log_freq=100)
    
    # scaler = torch.amp.GradScaler(enabled=train_config['amp'].amp)
    scaler = torch.cuda.amp.GradScaler(enabled=train_config['amp'])
    
    # Check if using zero-inflated model
    use_zero_inflated = train_config['model'].get('use_zero_inflated', False)
    
    cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean').to(local_rank)
    
    if use_zero_inflated:
        from loss.loss import ZeroInflatedLoss
        # Get individual loss weights from config
        bce_weight = train_config['task_weight'].get('zero_bce', 1.0)
        mse_weight = train_config['task_weight'].get('value_mse', 1.0)
        value_loss_fn = ZeroInflatedLoss(
            bce_weight=bce_weight,
            mse_weight=mse_weight
        ).to(local_rank)
    else:
        # Original mvc loss (for backward compatibility)
        mvc_loss_fn = MaskedMSELoss().to(local_rank)
        bce_weight = 0.0
        mse_weight = 0.0
    
    mvc_weight = train_config['task_weight']['mvc']  # Can be set to 0 for zero-inflated
    cell_type_weight = train_config['task_weight']['cell_type']
    
    softmax = nn.Softmax(dim=-1)
    
    steps = 0
    if train_config['model']['pretrained'] is not None:
        if is_master:
            logger.info('Loading pretrained model from: {}'.format(train_config['model']['pretrained']))
        checkpoint = torch.load(train_config['model']['pretrained'], map_location=device)
        
        # # do not load value_decoder parameters
        pretrained_dict = {k: v for k, v in checkpoint['model'].items() if 'value_decoder' not in k and 'mvc_decoder' not in k and 'batch_emb' not in k and 'cls_decoder' not in k}
        model_dict = model.module.state_dict()
        model_dict.update(pretrained_dict)
        model.module.load_state_dict(model_dict)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        
        # scheduler.load_state_dict(checkpoint['scheduler'])
        # scaler.load_state_dict(checkpoint['scaler'])
        if resume:
            start_epoch = checkpoint['epoch'] + 1
            steps = checkpoint['steps']
        del checkpoint
        del pretrained_dict
        gc.collect()
    
    dist.barrier()
    if is_master:
        logger.info('Start finetuning from epoch: {}, steps: {}'.format(start_epoch, steps))
    for i in range(start_epoch, start_epoch + EPOCHS):
        train_loader.sampler.set_epoch(i)
        
        if is_master:
            logger.info('Training with {} samples, steps: {}'.format(len(train_loader.dataset), len(train_loader)))
        model.train()
        dist.barrier()
        running_loss = {'mvc': 0.0, 'cell': 0.0, 'total': 0.0}
        cum_acc_cell = 0.0
        cum_acc_zero = 0.0  # Accumulator for zero prediction accuracy
        for index, batch in enumerate(train_loader):
            index += 1
            steps += 1
            rna_values = batch['rna_values'].to(device)
            rna_ids = batch['rna_ids'].to(device)
            atac_ids = batch['atac_ids'].to(device)
            cell_ids = batch['cell_ids'].to(device)
            batch_ids = batch['batch_ids'].to(device)
            rna_chrs = batch['rna_chrs'].to(device)
            atac_chrs = batch['atac_chrs'].to(device)
            
            padding_positions = atac_ids.eq(atac_vocab[pad['token']])
            rna_non_pad = rna_ids.ne(rna_vocab[pad['token']])
            
            if index % GRADIENT_ACCUMULATION != 0 and index != len(train_loader):
                with model.no_sync():
                    with torch.cuda.amp.autocast(enabled=train_config['amp'], dtype= torch.bfloat16):
                        # finetue using all expression values, do not mask
                        output = model(atac = atac_ids, rna = rna_ids, src_key_padding_mask = padding_positions, batch_id = batch_ids, rna_chrs = rna_chrs, atac_chrs = atac_chrs)

                        if use_zero_inflated:
                            value_output = output['value_pred']  # Dict with 'regression' and 'zero_logits'
                            value_loss, bce_loss, mse_loss = value_loss_fn(
                                value_output['regression'],
                                value_output['zero_logits'],
                                rna_values.float(),
                                mask=rna_non_pad
                            )
                            # mvc_weight acts as overall value_loss weight
                            loss = value_loss * mvc_weight + cross_entropy_loss(output['cell_pred'], cell_ids) * cell_type_weight
                            
                            # Calculate weighted losses for logging
                            weighted_bce = bce_loss.item() * bce_weight
                            weighted_mse = mse_loss.item() * mse_weight
                            
                            running_loss['value'] = running_loss.get('value', 0.0) + value_loss.item()
                            running_loss['bce_weighted'] = running_loss.get('bce_weighted', 0.0) + weighted_bce
                            running_loss['mse_weighted'] = running_loss.get('mse_weighted', 0.0) + weighted_mse
                            running_loss['cell'] = running_loss.get('cell', 0.0) + cross_entropy_loss(output['cell_pred'], cell_ids).item()
                            running_loss['total'] = running_loss.get('total', 0.0) + loss.item()
                            
                            # Calculate zero prediction accuracy
                            zero_probs = torch.sigmoid(value_output['zero_logits'])
                            zero_pred = (zero_probs > 0.5).float()  # Binary prediction: 1 if prob > 0.5
                            is_zero_target = (rna_values == 0).float()
                            zero_correct = ((zero_pred == is_zero_target) * rna_non_pad.float()).sum()
                            zero_total = rna_non_pad.float().sum()
                            cum_acc_zero += (zero_correct / zero_total).item()
                        else:
                            # Original path: mvc_pred uses binned values
                            mvc_loss = mvc_loss_fn(output['value_pred'], rna_values.float(), mask=rna_non_pad) * mvc_weight
                            cell_loss = cross_entropy_loss(output['cell_pred'], cell_ids) * cell_type_weight
                            loss = mvc_loss + cell_loss
                            
                            running_loss['mvc'] = running_loss.get('mvc', 0.0) + mvc_loss.item()
                            running_loss['cell'] = running_loss.get('cell', 0.0) + cell_loss.item()
                            running_loss['total'] = running_loss.get('total', 0.0) + loss.item()
                        
                        loss = loss / GRADIENT_ACCUMULATION
                    scaler.scale(loss).backward()
            else:
                with torch.cuda.amp.autocast(enabled=train_config['amp'], dtype= torch.bfloat16):
                    output = model(atac = atac_ids, rna = rna_ids, src_key_padding_mask = padding_positions, batch_id = batch_ids, rna_chrs = rna_chrs, atac_chrs = atac_chrs)
                    
                    if use_zero_inflated:
                        value_output = output['value_pred']  # Dict with 'regression' and 'zero_logits'
                        value_loss, bce_loss, mse_loss = value_loss_fn(
                            value_output['regression'],
                            value_output['zero_logits'],
                            rna_values.float(),
                            mask=rna_non_pad
                        )
                        cell_loss = cross_entropy_loss(output['cell_pred'], cell_ids) * cell_type_weight
                        loss = value_loss * mvc_weight + cell_loss
                        
                        # Calculate weighted losses for logging
                        weighted_bce = bce_loss.item() * bce_weight
                        weighted_mse = mse_loss.item() * mse_weight
                        
                        running_loss['value'] = running_loss.get('value', 0.0) + value_loss.item()
                        running_loss['bce'] = running_loss.get('bce', 0.0) + bce_loss.item()
                        running_loss['mse'] = running_loss.get('mse', 0.0) + mse_loss.item()
                        running_loss['bce_weighted'] = running_loss.get('bce_weighted', 0.0) + weighted_bce
                        running_loss['mse_weighted'] = running_loss.get('mse_weighted', 0.0) + weighted_mse
                        running_loss['cell'] = running_loss.get('cell', 0.0) + cell_loss.item()
                        running_loss['total'] = running_loss.get('total', 0.0) + loss.item()
                        
                        # Calculate zero prediction accuracy
                        zero_probs = torch.sigmoid(value_output['zero_logits'])
                        zero_pred = (zero_probs > 0.5).float()  # Binary prediction: 1 if prob > 0.5
                        is_zero_target = (rna_values == 0).float()
                        zero_correct = ((zero_pred == is_zero_target) * rna_non_pad.float()).sum()
                        zero_total = rna_non_pad.float().sum()
                        batch_zero_acc = (zero_correct / zero_total).item()
                        cum_acc_zero += batch_zero_acc
                        
                        if is_master:
                            tb_logger.add_scalar('train/value_loss', value_loss.item(), steps)
                            tb_logger.add_scalar('train/value_bce_weighted', weighted_bce, steps)
                            tb_logger.add_scalar('train/value_mse_weighted', weighted_mse, steps)
                            tb_logger.add_scalar('train/cell_loss_weighted', cell_loss.item(), steps)
                            tb_logger.add_scalar('train/total_loss', loss.item(), steps)
                            tb_logger.add_scalar('train/zero_accuracy', batch_zero_acc * 100, steps)
                            
                            # Log to wandb
                            wandb.log({
                                'train/value_loss': value_loss.item(),
                                'train/value_bce_weighted': weighted_bce,
                                'train/value_mse_weighted': weighted_mse,
                                'train/cell_loss_weighted': cell_loss.item(),
                                'train/total_loss': loss.item(),
                                'train/zero_accuracy': batch_zero_acc * 100,
                                'train/step': steps,
                                'train/epoch': i
                            }, step=steps)
                            
                            logger.info(f'Epoch: {i} | Step: {index} | Value: {value_loss:.4f} | BCE: {weighted_bce:.4f} | MSE: {weighted_mse:.4f} | Cell: {cell_loss:.4f} | Zero Acc: {batch_zero_acc*100:.2f}% | Total: {loss:.4f}')
                    else:
                        # Original path: mvc_pred uses binned values
                        mvc_loss = mvc_loss_fn(output['value_pred'], rna_values.float(), mask=rna_non_pad) * mvc_weight
                        cell_loss = cross_entropy_loss(output['cell_pred'], cell_ids) * cell_type_weight
                        loss = mvc_loss + cell_loss
                        
                        running_loss['mvc'] = running_loss.get('mvc', 0.0) + mvc_loss.item()
                        running_loss['cell'] = running_loss.get('cell', 0.0) + cell_loss.item()
                        running_loss['total'] = running_loss.get('total', 0.0) + loss.item()
                        
                        if is_master:
                            tb_logger.add_scalar('train/mvc_loss', mvc_loss.item(), steps)
                            tb_logger.add_scalar('train/cell_loss', cell_loss.item(), steps)
                            tb_logger.add_scalar('train/total_loss', loss.item(), steps)
                            
                            # Log to wandb
                            wandb.log({
                                'train/mvc_loss': mvc_loss.item(),
                                'train/cell_loss': cell_loss.item(),
                                'train/total_loss': loss.item(),
                                'train/step': steps,
                                'train/epoch': i
                            }, step=steps)
                            
                            logger.info(f'Epoch: {i} | Step: {index} | MVC Loss: {mvc_loss:.4f} | Cell Type Loss: {cell_loss:.4f} | Total Loss: {loss:.4f}')
                    
                    loss = loss / GRADIENT_ACCUMULATION
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e2))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            # cell type accuracy
            type_pred = softmax(output['cell_pred'])
            type_pred = type_pred.argmax(dim=-1)
            cum_acc_cell += (type_pred.eq(cell_ids)).sum().item() / len(cell_ids)
            
        cum_acc_cell = 100 * cum_acc_cell / index
        cum_acc_cell = get_reduced(cum_acc_cell, local_rank, 0, world_size)
        
        if use_zero_inflated:
            cum_acc_zero = 100 * cum_acc_zero / index
            cum_acc_zero = get_reduced(cum_acc_zero, local_rank, 0, world_size)
        
        for key in running_loss:
            running_loss[key] = running_loss[key] / index
            running_loss[key] = get_reduced(running_loss[key], local_rank, 0, world_size)
        if is_master:
                # Log epoch-level metrics to wandb
                if use_zero_inflated:
                    wandb.log({
                        'train_epoch/value_loss': running_loss.get("value", 0.0),
                        'train_epoch/bce_weighted': running_loss.get("bce_weighted", 0.0),
                        'train_epoch/mse_weighted': running_loss.get("mse_weighted", 0.0),
                        'train_epoch/cell_loss_weighted': running_loss.get("cell", 0.0),
                        'train_epoch/total_loss': running_loss.get("total", 0.0),
                        'train_epoch/cell_accuracy': cum_acc_cell,
                        'train_epoch/zero_accuracy': cum_acc_zero,
                        'train_epoch/epoch': i,
                        'train_epoch/learning_rate': optimizer.param_groups[0]['lr']
                    }, step=steps)
                    
                    logger.info(f'Epoch: {i} | Value: {running_loss.get("value", 0.0):.4f} | BCE: {running_loss.get("bce_weighted", 0.0):.4f} | MSE: {running_loss.get("mse_weighted", 0.0):.4f} | Cell: {running_loss.get("cell", 0.0):.4f} | Zero Acc: {cum_acc_zero:.2f}% | Cell Acc: {cum_acc_cell:.2f}% | Total: {running_loss.get("total", 0.0):.4f}')
                else:
                    wandb.log({
                        'train_epoch/mvc_loss': running_loss.get("mvc", 0.0),
                        'train_epoch/cell_loss': running_loss.get("cell", 0.0),
                        'train_epoch/total_loss': running_loss.get("total", 0.0),
                        'train_epoch/cell_accuracy': cum_acc_cell,
                        'train_epoch/epoch': i,
                        'train_epoch/learning_rate': optimizer.param_groups[0]['lr']
                    }, step=steps)
                    
                    logger.info(f'Epoch: {i} | MVC Loss: {running_loss.get("mvc", 0.0):.4f} | Cell Type Loss: {running_loss.get("cell", 0.0):.4f} | Total Loss: {running_loss.get("total", 0.0):.4f} | Cell Type Accuracy: {cum_acc_cell:.2f}')
        dist.barrier()
        scheduler.step()
        # del train_set, train_sampler, train_loader

        if i % valid_config['freq'] == 0:
            if is_master:
                logger.info('#### Validation ####')
            model.eval()
            dist.barrier()
            running_loss = {'mvc': 0.0, 'cell': 0.0, 'total': 0.0}
            
            cum_acc_cell = 0.0
            cum_acc_zero = 0.0  # Accumulator for validation zero prediction accuracy

            with torch.no_grad():
                for index, batch in enumerate(val_loader):
                    index += 1
                    
                    rna_values = batch['rna_values'].to(device)
                    rna_ids = batch['rna_ids'].to(device)
                    atac_ids = batch['atac_ids'].to(device)
                    cell_ids = batch['cell_ids'].to(device)
                    batch_ids = batch['batch_ids'].to(device)
                    rna_chrs = batch['rna_chrs'].to(device)
                    atac_chrs = batch['atac_chrs'].to(device)

                    padding_positions = atac_ids.eq(atac_vocab[pad['token']])
                    rna_non_pad = rna_ids.ne(rna_vocab[pad['token']])
                    with torch.cuda.amp.autocast(enabled=train_config['amp'], dtype= torch.bfloat16):
                        output = model(atac = atac_ids, rna = rna_ids, src_key_padding_mask = padding_positions, batch_id = batch_ids, rna_chrs = rna_chrs, atac_chrs = atac_chrs)
                        
                        if use_zero_inflated:
                            value_output = output['value_pred']  # Dict with 'regression' and 'zero_logits'
                            value_loss, bce_loss, mse_loss = value_loss_fn(
                                value_output['regression'],
                                value_output['zero_logits'],
                                rna_values.float(),
                                mask=rna_non_pad
                            )
                            cell_loss = cross_entropy_loss(output['cell_pred'], cell_ids) * cell_type_weight
                            loss = value_loss * mvc_weight + cell_loss
                            
                            running_loss['value'] = running_loss.get('value', 0.0) + value_loss.item()
                            running_loss['bce'] = running_loss.get('bce', 0.0) + bce_loss.item()
                            running_loss['mse'] = running_loss.get('mse', 0.0) + mse_loss.item()
                            running_loss['cell'] = running_loss.get('cell', 0.0) + cell_loss.item()
                            running_loss['total'] = running_loss.get('total', 0.0) + loss.item()
                            
                            # Calculate zero prediction accuracy
                            zero_probs = torch.sigmoid(value_output['zero_logits'])
                            zero_pred = (zero_probs > 0.5).float()
                            is_zero_target = (rna_values == 0).float()
                            zero_correct = ((zero_pred == is_zero_target) * rna_non_pad.float()).sum()
                            zero_total = rna_non_pad.float().sum()
                            cum_acc_zero += (zero_correct / zero_total).item()
                        else:
                            # Original path: mvc_pred uses binned values
                            mvc_loss = mvc_loss_fn(output['value_pred'], rna_values.float(), mask=rna_non_pad) * mvc_weight
                            cell_loss = cross_entropy_loss(output['cell_pred'], cell_ids) * cell_type_weight
                            loss = mvc_loss + cell_loss
                            
                            running_loss['mvc'] = running_loss.get('mvc', 0.0) + mvc_loss.item()
                            running_loss['cell'] = running_loss.get('cell', 0.0) + cell_loss.item()
                            running_loss['total'] = running_loss.get('total', 0.0) + loss.item()
                    
                    type_pred = softmax(output['cell_pred'])
                    type_pred = type_pred.argmax(dim=-1)
                    cum_acc_cell += (type_pred.eq(cell_ids)).sum().item() / len(cell_ids)
                    
                    # break   
            for key in running_loss:
                running_loss[key] = running_loss[key] / index
                running_loss[key] = get_reduced(running_loss[key], local_rank, 0, world_size)
            cum_acc_cell = 100 * cum_acc_cell / index
            cum_acc_cell = get_reduced(cum_acc_cell, local_rank, 0, world_size)
            
            if use_zero_inflated:
                cum_acc_zero = 100 * cum_acc_zero / index
                cum_acc_zero = get_reduced(cum_acc_zero, local_rank, 0, world_size)
            
            # del val_set, val_sampler, val_loader
            if is_master:
                # Log validation metrics to wandb
                if use_zero_inflated:
                    wandb.log({
                        'val/value_loss': running_loss.get("value", 0.0),
                        'val/value_bce_weighted': running_loss.get("bce_weighted", 0.0),
                        'val/value_mse_weighted': running_loss.get("mse_weighted", 0.0),
                        'val/cell_loss_weighted': running_loss.get("cell", 0.0),
                        'val/total_loss': running_loss.get("total", 0.0),
                        'val/cell_accuracy': cum_acc_cell,
                        'val/zero_accuracy': cum_acc_zero,
                        'val/epoch': i
                    }, step=steps)
                    
                    logger.info(f'Value: {running_loss.get("value", 0.0):.4f} | BCE: {running_loss.get("bce_weighted", 0.0):.4f} | MSE: {running_loss.get("mse_weighted", 0.0):.4f} | Cell: {running_loss.get("cell", 0.0):.4f} | Zero Acc: {cum_acc_zero:.2f}% | Cell Acc: {cum_acc_cell:.2f}% | Total: {running_loss.get("total", 0.0):.4f}')
                else:
                    wandb.log({
                        'val/mvc_loss': running_loss.get("mvc", 0.0),
                        'val/cell_loss': running_loss.get("cell", 0.0),
                        'val/total_loss': running_loss.get("total", 0.0),
                        'val/cell_accuracy': cum_acc_cell,
                        'val/epoch': i
                    }, step=steps)
                    
                    logger.info(f'MVC Loss: {running_loss.get("mvc", 0.0):.4f} | Cell Type Loss: {running_loss.get("cell", 0.0):.4f} | Total Loss: {running_loss.get("total", 0.0):.4f} | Cell Type Accuracy: {cum_acc_cell:.2f}')
                
        if is_master and i % save_ckpt_freq == 0:
            save_ckpt(i, steps,  model, optimizer, scheduler, scaler, running_loss["total"], task_name, ckpt_dir)
    
    # Finish wandb run
    if is_master:
        wandb.finish()

if __name__ == '__main__':
    main()
    