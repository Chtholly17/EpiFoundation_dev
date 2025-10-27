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
from data.dataloader import *
from tokenizer import GeneVocab
import scanpy as sc
import anndata as ad
from utils import *
from memory_profiler import profile

import yaml

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default='./configs/eval/baseline.yml', help='Config file.')
parser.add_argument("--backend", type=str, default='flash', help='Fast Transformer backend.')
args = parser.parse_args()

# @profile(precision=4, stream=open("memory_profiler.log", "w+"))
def main():
    # read and parse config file
    local_rank = int(os.environ["LOCAL_RANK"])
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    
    train_config = config['train']
    data_config = config['data']
    vocab_config = config['vocab']
    task_name = config['task_name']
    task_floder = './result/{}'.format(task_name)


    
    random_seed = train_config['seed']

    LEARNING_RATE = float(train_config['lr'])

    model_name = train_config['model']['encoder']

    # special tokens
    pad = vocab_config['special_tokens']['pad']
    cls = vocab_config['special_tokens']['cls']
    
    # distibuted setting
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed_all(random_seed + torch.distributed.get_rank())
    is_master = (local_rank == 0)
    
    # init loggers
    logger = set_log(log_dir= os.path.join(task_floder, 'logs'))
    if is_master:
        logger.info(dict2str(config))
    
    
    rna_vocab = GeneVocab.from_file(vocab_config['rna_path'])
    atac_vocab = GeneVocab.from_file(vocab_config['atac_path'])
    cell_vocab = GeneVocab.from_file(vocab_config['cell_type_path'])
    batch_vocab = GeneVocab.from_file(vocab_config['batch_path'])
    chr_vocab = GeneVocab.from_file(vocab_config['chr_path'])
    if is_master:
        logger.info(f'Rna vocab size: {len(rna_vocab)}')
        logger.info(f'Atac vocab size: {len(atac_vocab)}')
        
    if is_master:
        logger.info('loading training data')

    if vocab_config['hvg_path'] is not None:
        # read the hvg csv file
        hvg_df = pd.read_csv(vocab_config['hvg_path'])
        # get the first column as the gene names
        hvg_genes = hvg_df.iloc[:, 0].tolist()
    else:
        hvg_genes = None

    
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
        # get_full_genes= True,
        hvg_list=hvg_genes,
    )
    gc.collect()

    # create non distributed loader for evaluation
    if is_master:
        # for evaluation, batch size should be 1
        val_non_dist_loader = DataLoader(val_set, batch_size= 1, shuffle=False, num_workers=4) 
    
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
        transformer_backend = args.backend,
        stage= 'value_finetune',
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
    
    # scaler = torch.amp.GradScaler(enabled=train_config['amp'].amp)
    scaler = torch.cuda.amp.GradScaler(enabled=train_config['amp'])
    
    # masked_mse_loss = MaskedMSELoss().to(local_rank)
    cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean').to(local_rank)
    atac_cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index = pad['value']).to(local_rank)
    
    softmax = nn.Softmax(dim=-1)
    
    steps = 0
    if train_config['model']['pretrained'] is not None:
        if is_master:
            logger.info('Loading pretrained model from: {}'.format(train_config['model']['pretrained']))
        checkpoint = torch.load(train_config['model']['pretrained'], map_location=device)
        
        # # do not load batch_emb and cls_decoder parameters (when finetuning on different dataset)
        pretrained_dict = {k: v for k, v in checkpoint['model'].items() if 'batch_emb' not in k and 'cls_decoder' not in k}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        del checkpoint
        gc.collect()
    
    dist.barrier()
    
    
    mse_loss = nn.MSELoss(reduction='mean').to(local_rank)
    
    if train_config['metric'] == True:
        if is_master:
            model.eval() 
            test_adata = sc.read_h5ad(data_config['test']['rna_path'])
            
            batch_labels = []
            expression_preds = []
            cell_labels = []
            
            cum_loss = 0.0
            tbar = tqdm(val_non_dist_loader, desc='Eval')
            for index, batch in enumerate(val_non_dist_loader):
                tbar.update(1)
                index += 1
                # if index > 10:
                #     break
                
                rna_values = batch['rna_values'].to(device)
                rna_ids = batch['rna_ids'].to(device)
                atac_ids = batch['atac_ids'].to(device)
                cell_ids = batch['cell_ids'].to(device)
                batch_ids = batch['batch_ids'].to(device)
                rna_chrs = batch['rna_chrs'].to(device)
                atac_chrs = batch['atac_chrs'].to(device)
                
                padding_positions = atac_ids.eq(atac_vocab[pad['token']])
                with torch.cuda.amp.autocast(enabled=train_config['amp'], dtype= torch.bfloat16):
                    output = model(atac = atac_ids, rna = rna_ids, src_key_padding_mask = padding_positions, batch_id = batch_ids, rna_chrs = rna_chrs, atac_chrs = atac_chrs)
                    
                # Handle zero-inflated model inference
                use_zero_inflated = train_config['model'].get('use_zero_inflated', False)
                if use_zero_inflated:
                    value_output = output['value_pred']
                    zero_probs = torch.sigmoid(value_output['zero_logits'])
                    threshold = train_config['model'].get('zero_threshold', 0.5)
                    
                    # If p0 > threshold, predict 0; otherwise use regression output
                    pred_value = torch.where(
                        zero_probs > threshold,
                        torch.zeros_like(value_output['regression']),
                        value_output['regression']
                    )
                    
                    # log the number of zeros here
                    num_zeros = pred_value[pred_value == 0].shape[0]
                    logger.info(f'Number of zeros: {num_zeros}')
                else:
                    pred_value = output['value_pred']
                rna_value = rna_values
                
                with torch.no_grad():
                    current_loss = mse_loss(pred_value, rna_value)
                cum_loss += current_loss.item()
                
                
                # print("GT and Pred: ", cell_ids.cpu().numpy(), pred.cpu().numpy())
                pred_value = pred_value.float().detach().cpu().numpy()[0]
                rna_ids = rna_ids.detach().cpu().numpy()[0]
                # create a expression dict, where key is the gene id, value is initialized to 0
                if hvg_genes is not None:
                    # get all genes in the test_adata
                    full_gene_ids = val_non_dist_loader.dataset.get_full_genes_ids()
                    # convert to dict
                    full_gene_dict = {gene: -1 for gene in full_gene_ids}
                    # fill the dict with the values from pred, set the value of key rna_ids to pred
                    for i in range(len(rna_ids)):
                        full_gene_dict[rna_ids[i]] = pred_value[i]
                    # replace the pred_value[0] (which is a numpy array) with the full_gene_dict.values()
                    full_pred_value = np.array(list(full_gene_dict.values()))
                    full_pred_value = full_pred_value[np.newaxis, :]
                    expression_preds.append(full_pred_value) 
                else:
                    expression_preds.append(pred_value.float().detach().cpu().numpy())
                batch_labels.append(batch_ids.cpu().numpy())
                cell_labels.append(cell_ids.cpu().numpy())
            tbar.close()
            
            cum_loss /= len(val_non_dist_loader)
            logger.info(f'Validation Loss: {cum_loss}')
            
            
            
            # concatenate the results and transform to numpy array
            cell_labels = np.concatenate(cell_labels)
            # embeddings = np.concatenate(embeddings)
            batch_labels = np.concatenate(batch_labels)
            data_cell_ids = np.array(cell_vocab(test_adata.obs['annot'].tolist()))
            data_batch_ids = test_adata.obs['batch'].tolist()
            data_batch_ids = np.array(batch_vocab(data_batch_ids))
            assert np.all(data_cell_ids == cell_labels)
            assert np.all(data_batch_ids == batch_labels)
            # now cell_labels have shape (len(data_cell_ids), 1), convert to (len(data_cell_ids),)
            
            # save the expression_preds as a new layer in adata
            expression_preds = np.concatenate(expression_preds, axis=0)
            test_adata.layers['pred'] = expression_preds
            
            print("result adata: ", test_adata)
            
            test_adata.write_h5ad(os.path.join(task_floder,'pred_result.h5ad'))
                                
        

if __name__ == '__main__':
    main()
    