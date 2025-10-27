import scanpy as sc
from scanpy import AnnData
import scipy
import os
from model import logger
import numpy as np
from data.preprocess import Preprocessor
from tokenizer import GeneVocab
from data.dataloader import *
import yaml
from tqdm import tqdm

kindy_bmmc_file = ['GSM7321078_GSM7321079', 'GSM7321082_GSM7321083', 'GSM7321084_GSM7321085', 'GSM7321074_GSM7321075', 'GSM7321080_GSM7321081', 'GSM7321076_GSM7321077',
                   'GSM5828468_GSM5828481', 'GSM5828469_GSM5828482', 'GSM5828470_GSM5828483', 'GSM5828471_GSM5828484', 'GSM5828472_GSM5828485', 'GSM5828473_GSM5828486', 
                   'GSM5828474_GSM5828487', 'GSM5828475_GSM5828488', 'GSM5828476_GSM5828489', 'GSM5828477_GSM5828490', 'GSM5828478_GSM5828491', 'GSM5828479_GSM5828492', 
                   'GSM5828480_GSM5828493']
# kindy_bmmc_file = ['GSM7321078_GSM7321079', 'GSM7321082_GSM7321083']

def merge_files():
    data_path = '/home/jwu418/workspace/data/ours/raw/rna/'
    atac_path = '/home/jwu418/workspace/data/ours/raw/atac_binary/'
    # get the file names, and get the first 10 files
    files = os.listdir(data_path)
    files.sort()
    # train_files = files[9:10]
    # test_files = files[11:12]
    # valid_files = files[13:14]
    # read files
    train_data = []
    atac_train_data = []
    # test_data = []
    # atac_test_data = []
    # valid_data = []
    # atac_valid_data = []
    
    for file in kindy_bmmc_file:
        print('Reading:', file)
        file = file + '.h5ad'
        adata = sc.read(data_path + file)
        atac_data = sc.read(atac_path + file)
        batch_label = file.split('.')[0]
        # replace adata.X with adata.raw.X
        adata.X = adata.raw.X
        # delete the raw data
        adata.raw = None
        atac_data.raw = None
        
        min_value = adata.X.min()
        print('Min:', min_value)
        # add a obs column to store the batch label
        adata.obs['batch'] = batch_label
        atac_data.obs['batch'] = batch_label
        train_data.append(adata)
        atac_train_data.append(atac_data)
    
    # concatenate the data
    train_data = AnnData.concatenate(*train_data)
    atac_train_data = AnnData.concatenate(*atac_train_data)
    assert train_data.obs_names.tolist() == atac_train_data.obs_names.tolist(), 'The cell names should be the same.'
    print('Train Data:', train_data)
    
    # save files
    train_data.write('/home/jwu418/workspace/data/ours/raw/bmmc_kidney_rna_new.h5ad')
    atac_train_data.write('/home/jwu418/workspace/data/ours/raw/bmmc_kidney_atac_new.h5ad')
    
    # for file in test_files:
    #     print('Reading:', file)
    #     adata = sc.read(data_path + file)
    #     atac_data = sc.read(atac_path + file)
    #     batch_label = file.split('.')[0]
    #     # add a obs column to store the batch label
    #     adata.obs['batch'] = batch_label
    #     atac_data.obs['batch'] = batch_label
    #     test_data.append(adata)
    #     atac_test_data.append(atac_data)
    # # concatenate the data
    # test_data = AnnData.concatenate(*test_data)
    # atac_test_data = AnnData.concatenate(*atac_test_data)
    # print('Test Data:', test_data)
    
    # for file in valid_files:
    #     print('Reading:', file)
    #     adata = sc.read(data_path + file)
    #     atac_data = sc.read(atac_path + file)
    #     batch_label = file.split('.')[0]
    #     # add a obs column to store the batch label
    #     adata.obs['batch'] = batch_label
    #     atac_data.obs['batch'] = batch_label
    #     valid_data.append(adata)
    #     atac_valid_data.append(atac_data)
    # # concatenate the data
    # valid_data = AnnData.concatenate(*valid_data)
    # atac_valid_data = AnnData.concatenate(*atac_valid_data)
    # print('Valid Data:', valid_data)
    
    # preprocess_config = {
    #     'use_key': 'X',
    #     'filter_gene_by_counts': False,
    #     'filter_cell_by_counts': False,
    #     'normalize_total': 1e4,
    #     'result_normed_key': 'X_normed',
    #     'log1p': False,
    #     'result_log1p_key': 'X_log1p',
    #     'subset_hvg': False,
    #     'hvg_use_key': None,
    #     'hvg_flavor': 'seurat_v3',
    #     'binning': 2,
    #     'result_binned_key': 'X_binned',
    #     'batch_key': 'batch',
    # }
    
    # processor = Preprocessor(use_key=preprocess_config['use_key'],
    #                                 filter_gene_by_counts=preprocess_config['filter_gene_by_counts'],
    #                                 filter_cell_by_counts=preprocess_config['filter_cell_by_counts'],
    #                                 normalize_total=preprocess_config['normalize_total'],
    #                                 result_normed_key=preprocess_config['result_normed_key'],
    #                                 log1p=preprocess_config['log1p'],
    #                                 result_log1p_key=preprocess_config['result_log1p_key'],
    #                                 subset_hvg=preprocess_config['subset_hvg'],
    #                                 hvg_use_key=preprocess_config['hvg_use_key'],
    #                                 hvg_flavor=preprocess_config['hvg_flavor'],
    #                                 binning=preprocess_config['binning'],
    #                                 result_binned_key=preprocess_config['result_binned_key'])
    
    # output_name = '10_samples_rna_binned.h5ad'
    # atac_output_name = '10_samples_atac.h5ad'
    # processor(train_data, batch_key= preprocess_config['batch_key'])
    # train_data.write('/home/jwu418/workspace/data/ours/train/{}'.format(output_name))
    # atac_train_data.write('/home/jwu418/workspace/data/ours/train/{}'.format(atac_output_name))
    
    # processor(test_data, batch_key= preprocess_config['batch_key'])
    # test_data.write('/home/jwu418/workspace/data/ours/test/{}'.format(output_name))
    # atac_test_data.write('/home/jwu418/workspace/data/ours/test/{}'.format(atac_output_name))
    
    # processor(valid_data, batch_key= preprocess_config['batch_key'])
    # valid_data.write('/home/jwu418/workspace/data/ours/valid/{}'.format(output_name))
    # atac_valid_data.write('/home/jwu418/workspace/data/ours/valid/{}'.format(atac_output_name))
    

def divide_data(data: AnnData, radio: dict = {'train': 0.9, 'test': 0.05, 'valid': 0.05}):
    '''
    Divide the data into train, test, valid.
    return 3 AnnData object.
    '''
    assert sum(radio.values()) == 1.0, 'The sum of radio should be 1.0'
    data = data.copy()
    # get the number of cells
    n_cells = data.shape[0]
    # get the index of cells
    idx = np.arange(n_cells)
    np.random.shuffle(idx)
    # get the number of cells for each part
    n_train = int(n_cells * radio['train'])
    n_test = int(n_cells * radio['test'])
    n_valid = int(n_cells * radio['valid'])
    # divide the data
    train_data = data[idx[:n_train]]
    test_data = data[idx[n_train:n_train+n_test]]
    valid_data = data[idx[n_train+n_test:]]
    
    return train_data, test_data, valid_data
    
'''
    Meeting log: for gene data, we should normalize the expression data first (sc.pp.normalize_total(adata, target_sum=1e4))
    For ATAC, we should use the raw data directly.
'''
def preprocess():
    
    preprocess_config = {
        'path': '/home/jwu418/workspace/data/ours/',
        'raw_data': 'pbmc_rna_s1.h5ad',
        'use_key': 'X',
        'filter_gene_by_counts': False,
        'filter_cell_by_counts': False,
        'normalize_total': False,
        'result_normed_key': 'X_normed',
        'log1p': False,
        'result_log1p_key': 'X_log1p',
        'subset_hvg': False,
        'hvg_use_key': None,
        'hvg_flavor': 'seurat_v3',
        'binning': [2],
        'result_binned_key': 'X_binned',
        'batch_key': 'batch',
        'output_name': 'pbmc_rna_s1',
    }
    file = '{}/raw/{}'.format(preprocess_config['path'], preprocess_config['raw_data'])
    adata = sc.read_h5ad(file)
    # devide data into train, test, valid. with 0.8,0.1,0.1
    # adata._raw._var.rename(columns={'_index': 'genes'}, inplace=True)
    print(adata)
    
    train_data, test_data, valid_data = divide_data(adata)
    for binning in preprocess_config['binning']:
        logger.info('Binning: {}'.format(binning))
        processor = Preprocessor(use_key=preprocess_config['use_key'],
                                    filter_gene_by_counts=preprocess_config['filter_gene_by_counts'],
                                    filter_cell_by_counts=preprocess_config['filter_cell_by_counts'],
                                    normalize_total=preprocess_config['normalize_total'],
                                    result_normed_key=preprocess_config['result_normed_key'],
                                    log1p=preprocess_config['log1p'],
                                    result_log1p_key=preprocess_config['result_log1p_key'],
                                    subset_hvg=preprocess_config['subset_hvg'],
                                    hvg_use_key=preprocess_config['hvg_use_key'],
                                    hvg_flavor=preprocess_config['hvg_flavor'],
                                    binning=binning,
                                    result_binned_key=preprocess_config['result_binned_key'])
        #  Setting from scGPT:
        # preprocessor = Preprocessor(
        #     use_key="X",  # the key in adata.layers to use as raw data
        #     filter_gene_by_counts=3,  # step 1
        #     filter_cell_by_counts=False,  # step 2
        #     normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
        #     result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        #     log1p=data_is_raw,  # 4. whether to log1p the normalized data
        #     result_log1p_key="X_log1p",
        #     subset_hvg=n_hvg,  # 5. whether to subset the raw data to highly variable genes
        #     hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        #     binning=config.n_bins,  # 6. whether to bin the raw data and to what number of bins
        #     result_binned_key="X_binned",  # the key in adata.layers to store the binned data
        # )
        
        
        
        output_name = f'{preprocess_config["output_name"]}_binning_{binning}'
        
        logger.info('Preprocessing Train Data')
        processor(train_data, batch_key= preprocess_config['batch_key'])
        print(train_data)
        train_data.write('{}/train/{}.h5ad'.format(preprocess_config['path'], output_name))

        logger.info('Preprocessing test Data')
        processor(test_data, batch_key= preprocess_config['batch_key'])
        print(test_data)
        test_data.write('{}/test/{}.h5ad'.format(preprocess_config['path'], output_name))
        
        logger.info('Preprocessing valid Data')
        processor(valid_data, batch_key= preprocess_config['batch_key'])
        print(valid_data)
        valid_data.write('{}/valid/{}.h5ad'.format(preprocess_config['path'], output_name))
    
    
    # save preprocess config as a yml file
    with open('/home/jwu418/workspace/data/ours/configs/{}.yml'.format(output_name), 'w') as file:
        yaml.dump(preprocess_config, file)
        
        
def reduce_data():
    path = '/home/jwu418/workspace/data/ours'
    rna_file = 'pbmc_rna_s1_binning_2.h5ad'
    
    stage = ['test', 'valid', 'train']
    for s in stage:
        adata = sc.read_h5ad('{}/{}/{}'.format(path, s, rna_file))
        print('Before:', adata)
        # remove the adata.raw
        adata.raw = None
        # save the X_binned as X
        adata.X = adata.layers['X_binned']
        # save adata.X as sparse matrix
        adata.X = scipy.sparse.csr_matrix(adata.X)
        # remove the X_binned layer
        adata.layers.pop('X_binned')
        # save the data as a new file
        adata.write('{}/{}/{}'.format(path, s, 'pbmc_rna_s1_binning_2_reduced.h5ad'))
        
def get_pair_data():
    path = '/home/jwu418/workspace/data/ours'
    rna_file = 'pbmc_rna_s1_binning_2.h5ad'
    atac_file = 'raw/pbmc_atac_s1.h5ad'
    
    output_name = 'pbmc_rna_s1_atac_paired.h5ad'
    
    stage = ['test', 'valid', 'train']
    
    atac = sc.read_h5ad('{}/{}'.format(path, atac_file))
    # breakpoint()
    for s in stage:
        rna = sc.read_h5ad('{}/{}/{}'.format(path, s, rna_file), backed='r')
        print('rna:', rna)
        # get the cell name of rna data

        rna_cell_name = rna.obs_names.tolist()
        # find the corresponding cell in atac data
        atac_cell = atac[rna_cell_name]
        # atac_cell._raw._var.rename(columns={'_index': 'peaks'}, inplace=True)
        # save the atac data as a new file
        atac_cell.write('{}/{}/{}'.format(path, s, output_name))
        print('atac cell:', atac_cell)
        
def generate_chr_vocab():
    file = '/home/jwu418/workspace/data/ours/meta/genes.csv'
    import pandas as pd
    # read the 'seqnames' column
    df = pd.read_csv(file)
    chr_names = df['seqnames'].tolist()
    chr_names = list(set(chr_names))
    vocab = GeneVocab(gene_list_or_vocab=chr_names,
                      special_first= True,
                       specials=['<pad>', '<mask>','<cls>', '<eos>'])
    vocab.save_json('/home/jwu418/workspace/data/ours/chr_vocab.json')

    
def generate_vocab():
    file = '/home/jwu418/workspace/data/ours/raw/mini_atlas_atac.h5ad'
    adata = sc.read_h5ad(file)
    # adata._raw._var.rename(columns={'_index': 'features'}, inplace=True)
    
    # get the gene names
    gene_names = adata.var_names.tolist()
    vocab = GeneVocab(gene_list_or_vocab=gene_names,
                        special_first= True,
                        specials=['<pad>', '<mask>','<cls>', '<eos>'])
    vocab.save_json('/home/jwu418/workspace/data/ours/atac_vocab.json')
    
def generate_cell_type_vocab():
    file = '/home/jwu418/workspace/data/ours/raw/pbmc_rna_s1.h5ad'
    adata = sc.read_h5ad(file)
    print(adata)
    # get the gene names
    gene_names = adata.obs['batch'].tolist()
    
    # remove duplicates
    gene_names = list(set(gene_names))
    print(gene_names)
    # get number of cell types
    print('Number of cell types:', len(gene_names))
    vocab = GeneVocab(gene_list_or_vocab=gene_names)
    vocab.save_json('/home/jwu418/workspace/data/ours/vocab/pbmc_s1_batch_vocab.json')
    
def non_zero_numbs():
    '''
    Get the number of non-zero values in the data.
    '''
    file = '/home/jwu418/workspace/data/ours/raw/bmmc_atac.h5ad'
    adata = sc.read_h5ad(file)
    # traverse all cells
    non_zero = []
    for i in tqdm(range(adata.shape[0])):
        non_zero.append(len(adata.X[i].nonzero()[0]))
    # get the min, max, mean of the list
    non_zero = np.array(non_zero)
    print('min:', non_zero.min())
    print('max:', non_zero.max())
    print('mean:', non_zero.mean())
    print('median:', np.median(non_zero))

def adjust_bin(target_bin_num = 10):
    origin_file = ['/home/jwu418/workspace/data/ours/train/mini_atlas_rna_binned_binning_2.h5ad',
                    '/home/jwu418/workspace/data/ours/test/mini_atlas_rna_binned_binning_2.h5ad',
                    '/home/jwu418/workspace/data/ours/valid/mini_atlas_rna_binned_binning_2.h5ad',]
    preprocessor = Preprocessor(
        use_key="X", 
        filter_gene_by_counts=False, 
        filter_cell_by_counts=False, 
        normalize_total=1e4,
        result_normed_key="X_normed", 
        log1p=False,
        result_log1p_key="X_log1p",
        subset_hvg=False,
        hvg_flavor="seurat_v3",
        binning=target_bin_num,  
        result_binned_key="X_binned",  # the key in adata.layers to store the binned
    )
    for file in origin_file:
        print('Reading:', file)
        adata = sc.read_h5ad(file)
        # remove the X_binned layer
        adata.layers.pop('X_binned')
        # preprocess the data
        preprocessor(adata, batch_key='batch')
        
        # remove the adata.raw
        adata.raw = None
        # save the X_binned as X
        adata.X = adata.layers['X_binned']
        # save adata.X as sparse matrix
        adata.X = scipy.sparse.csr_matrix(adata.X)
        # remove the X_binned layer
        adata.layers.pop('X_binned')
        
        # save the data as a new file
        new_file = file.replace('mini_atlas_rna_binned_binning_2', f'mini_atlas_rna_binned_binning_{target_bin_num}')
        adata.write(new_file)
        
        
        
        
def finetune_data_division(preprocess_config):
    rna_adata = sc.read_h5ad('{}/raw/{}'.format(preprocess_config['path'], preprocess_config['raw_data']))
    atac_adata = sc.read_h5ad('{}/raw/{}'.format(preprocess_config['path'], preprocess_config['atac_data']))
    # devide data into train, test, valid. with 0.8,0.1,0.1
    # adata._raw._var.rename(columns={'_index': 'genes'}, inplace=True)
    print('Raw RNA data: {}'.format(rna_adata))
    print('Raw ATAC data: {}\n\n'.format(atac_adata))
    
    # generate vocab
    cell_types = rna_adata.obs['annot'].tolist()
    batch_names = rna_adata.obs['batch'].tolist()
    cell_types = list(set(cell_types))
    batch_names = list(set(batch_names))
    print('Cell types:', cell_types)
    print('Batch names:', batch_names)
    print('\n\n')
    
    cell_vocab = GeneVocab(gene_list_or_vocab=cell_types)
    batch_vocab = GeneVocab(gene_list_or_vocab=batch_names)
    cell_vocab.save_json('{}/vocab/fine_tune_cell_type_vocab.json'.format(preprocess_config['path']))
    batch_vocab.save_json('{}/vocab/fine_tune_batch_vocab.json'.format(preprocess_config['path']))
    
    rna_adata_set = {}
    rna_adata_set['train'], rna_adata_set['test'], rna_adata_set['valid'] = divide_data(rna_adata)
    output_name = f'{preprocess_config["output_name"]}_raw.h5ad'
    
    atac_name = f'{preprocess_config["output_atac_name"]}.h5ad'
    for s in ['train', 'test', 'valid']:
        # get paired atac data
        print('RNA data for {}: {}'.format(s, rna_adata_set[s]))
        rna_cell_name = rna_adata_set[s].obs_names.tolist()
        atac_cell = atac_adata[rna_cell_name]
        
        atac_cell.write('{}/{}/{}'.format(preprocess_config['path'], s, atac_name))
        rna_adata_set[s].write('{}/{}/{}'.format(preprocess_config['path'], s, output_name))
        

def finetune_process_pipeline():
    preprocess_config = {
        'path': '/home/jwu418/workspace/data/ours/',
        'raw_data': 'kidney_atlas_rna.h5ad',
        'atac_data': 'kidney_atlas_atac.h5ad',
        'use_key': 'X',
        'filter_gene_by_counts': False,
        'filter_cell_by_counts': False,
        'normalize_total': False,
        'result_normed_key': 'X_normed',
        'log1p': False,
        'result_log1p_key': 'X_log1p',
        'subset_hvg': False,
        'hvg_use_key': None,
        'hvg_flavor': 'seurat_v3',
        'binning': [2,10],
        'result_binned_key': 'X_binned',
        'batch_key': 'batch',
        'output_name': 'kidney_atlas_rna',
        'output_atac_name': 'kidney_atlas_atac',
    }
    
    finetune_data_division(preprocess_config)
    for binning in preprocess_config['binning']:
        processor = Preprocessor(use_key=preprocess_config['use_key'],
                                    filter_gene_by_counts=preprocess_config['filter_gene_by_counts'],
                                    filter_cell_by_counts=preprocess_config['filter_cell_by_counts'],
                                    normalize_total=preprocess_config['normalize_total'],
                                    result_normed_key=preprocess_config['result_normed_key'],
                                    log1p=preprocess_config['log1p'],
                                    result_log1p_key=preprocess_config['result_log1p_key'],
                                    subset_hvg=preprocess_config['subset_hvg'],
                                    hvg_use_key=preprocess_config['hvg_use_key'],
                                    hvg_flavor=preprocess_config['hvg_flavor'],
                                    binning=binning,
                                    result_binned_key=preprocess_config['result_binned_key'])
        for s in ['train', 'test', 'valid']:
            raw_data_name = f'{preprocess_config["output_name"]}_raw.h5ad'
            adata = sc.read_h5ad('{}/{}/{}'.format(preprocess_config['path'], s, raw_data_name))
            
            processor(adata, batch_key= preprocess_config['batch_key'])
            
            print("\n\nData with {} binning: {}".format(binning, adata))
            # reduce the data
            adata.raw = None
            adata.X = adata.layers['X_binned']
            adata.X = scipy.sparse.csr_matrix(adata.X)
            adata.layers.pop('X_binned')
            adata.write('{}/{}/{}_binning_{}.h5ad'.format(preprocess_config['path'], s, preprocess_config['output_name'], binning))
            
        
    

if __name__ == '__main__':
    # generate_cell_type_vocab()
    # reduce_data()
    # generate_cell_type_vocab()
    # finetune_process_pipeline()
    adjust_bin()
    # generate_cell_type_vocab()
    # generate_cell_type_vocab()
    # adjust_bin(100)
    # get_pair_data()
    # generate_chr_vocab()
    # preprocess()
    # generate_cell_type_vocab()
    # test_prepare_dataloader()
    # generate_vocab()
    
    
'''
Proceesing the data:
1. preprocess()
2. get_pair_data()
3. generate_vocab()
4. generate_cell_type_vocab()
5. reduce_data()
'''