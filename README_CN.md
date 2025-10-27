# EpiFoundation: 单细胞多组学基础模型

EpiFoundation是一个基于Transformer的单细胞多组学数据整合基础模型，专为学习配对的ATAC-seq和RNA-seq数据表征而设计。本仓库包含模型预训练、微调和评估的完整代码。

## 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [数据准备](#数据准备)
- [配置文件](#配置文件)
- [训练流程](#训练流程)
- [常见问题](#常见问题)

## 安装

### 1. 创建Python环境

```bash
# 使用conda（推荐）
conda create -n epifoundation python=3.9
conda activate epifoundation

# 或使用venv
python -m venv epifoundation
source epifoundation/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 安装PyTorch

根据CUDA版本安装对应的PyTorch：

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 快速开始

### 1. 数据预处理

```python
from data.preprocess import Preprocessor
import scanpy as sc

# 加载原始数据
adata = sc.read_h5ad('path/to/your/data.h5ad')

# 预处理配置
preprocess_config = {
    'path': '/path/to/output/',
    'raw_data': 'your_data.h5ad',
    'use_key': 'X',
    'normalize_total': 1e4,
    'binning': 52,  # 表达值分箱数
    'result_binned_key': 'X_binned',
    'output_name': 'processed_data'
}

# 执行预处理
preprocessor = Preprocessor(preprocess_config)
preprocessor.preprocess()
```

### 2. 预训练

```bash
# 单GPU
CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 --master_port 29502 \
    pretrain.py --config ./configs/pretrain/atac_cross_debug.yml

# 多GPU（8卡）
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node=8 --master_port 29502 \
    pretrain.py --config ./configs/pretrain/atac_cross_debug.yml
```

### 3. 微调

```bash
CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 --master_port 29502 \
    finetune.py --config ./configs/finetune/mini_atlas_10bins.yml
```

### 4. 评估

```bash
CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 --master_port 29502 \
    eval.py --config ./configs/eval/mini_atlas_rna_bmmc.yml
```

## 数据准备

### 数据格式

模型需要AnnData格式（`.h5ad`）的数据，包含：

- **RNA-seq数据**：基因表达矩阵（细胞×基因）
- **ATAC-seq数据**：染色质可及性矩阵（细胞×峰）
- **元数据**：细胞类型标签、批次信息等

### 预处理步骤

1. **加载原始数据**：从原始计数矩阵开始
2. **标准化**：将总计数标准化到10,000（默认）
3. **分箱**：将连续的表达值离散化为区间（如52个箱）
4. **创建词表**：生成基因和细胞类型词表

详细说明请查看 [DATA_PREPARATION.md](docs/DATA_PREPARATION.md)。

### 词表文件

需要创建以下词表文件：
- **RNA基因词表** (`rna_vocab.json`)
- **ATAC峰词表** (`atac_vocab.json`)
- **细胞类型词表** (`cell_type_vocab.json`)
- **批次词表** (`batch_vocab.json`)
- **染色体词表** (`chr_vocab.json`)
- **基因-染色体映射** (`gene2chr.json`)

## 配置文件

所有实验通过`configs/`目录中的YAML配置文件控制：

- `configs/pretrain/`：预训练配置
- `configs/finetune/`：微调配置
- `configs/eval/`：评估配置

### 配置文件示例

```yaml
task_name: my_pretrain_task

train:
  seed: 2002
  batch_size: 8
  lr: 1e-4
  epochs: 150
  gradient_accumulation_steps: 20
  amp: True  # 自动混合精度
  save_ckpt_freq: 20
  resume: False

  model:
    encoder: transformer
    pretrained: null
    embedding_dim: 512
    num_layers: 6
    head_num: 8
    dropout: 0.15
    atac_max_len: 12000
    rna_max_len: 8000
  
  task_weight:
    cell_type: 0.0  # 细胞类型分类损失权重
    mvc: 1.0        # 掩码值预测损失权重

data:
  bin_num: 2
  train:
    atac_path: /path/to/train/atac.h5ad
    rna_path: /path/to/train/rna.h5ad

vocab:
  rna_path: /path/to/rna_vocab.json
  atac_path: /path/to/atac_vocab.json
  cell_type_path: /path/to/cell_type_vocab.json
```

详细参数说明请查看 [CONFIGURATION_GUIDE.md](docs/CONFIGURATION_GUIDE.md)。

### 关键参数说明

#### 训练参数

- **batch_size**：每个GPU的批次大小
- **lr**：学习率（预训练默认：1e-4，微调：1e-5到5e-5）
- **epochs**：训练轮数
- **gradient_accumulation_steps**：梯度累积步数
- **amp**：自动混合精度（True/False）

#### 模型参数

- **encoder**：模型架构（'transformer', 'performer'）
- **pretrained**：预训练模型路径（微调时使用）
- **embedding_dim**：token嵌入维度
- **num_layers**：Transformer层数
- **head_num**：注意力头数
- **dropout**：Dropout概率

#### 任务权重

- **cell_type**：细胞类型分类损失权重
- **mvc**：掩码值预测损失权重

## 训练流程

### 预训练

使用配对的ATAC-seq和RNA-seq数据进行掩码值预测（MVC）预训练。

```bash
# 多GPU训练
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun \
    --nproc_per_node=8 \
    --master_port 29502 \
    pretrain.py \
    --config ./configs/pretrain/atac_cross_debug.yml
```

**输出**：
- 检查点保存在 `experiment/<task_name>/ckpts/`
- TensorBoard日志在 `experiment/<task_name>/logs/`
- 预训练模型：`pretrain.pth`

### 微调

将预训练模型适配到特定下游任务。

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun \
    --nproc_per_node=8 \
    --master_port 29502 \
    finetune.py \
    --config ./configs/finetune/mini_atlas_10bins.yml
```

**微调任务**：
1. **细胞类型分类**：从单细胞profile预测细胞类型
2. **掩码值预测**：继续预训练目标
3. **零膨胀回归**：预测连续表达值（处理零值）

**输出**：
- 微调模型保存在 `experiment/<task_name>/ckpts/finetuned.pth`

### 评估

在测试集上评估模型性能。

```bash
CUDA_VISIBLE_DEVICES="0" torchrun \
    --nproc_per_node=1 \
    --master_port 29502 \
    eval.py \
    --config ./configs/eval/mini_atlas_rna_bmmc.yml
```

**评估指标**：

细胞类型分类：
- 准确率
- F1分数
- 混淆矩阵

表达预测：
- Pearson相关系数
- Spearman相关系数
- MSE
- R²分数

## 模型架构

EpiFoundation基于Transformer编码器架构，包含：

1. **Token嵌入**：嵌入基因/峰ID和表达值
2. **位置编码**：编码序列位置信息
3. **Transformer编码器**：多层自注意力机制
4. **跨模态注意力**：整合ATAC和RNA模态信息
5. **任务特定头**：
   - 掩码值预测头
   - 细胞类型分类头
   - 零膨胀回归头（可选）

## 常见问题

### CUDA内存不足

**解决方案**：
- 减小batch_size
- 增加gradient_accumulation_steps
- 启用AMP（`amp: True`）
- 减小序列长度

```yaml
train:
  batch_size: 4  # 从8减到4
  gradient_accumulation_steps: 40  # 从20增到40
  amp: True
```

### 端口被占用

**解决方案**：更改`--master_port`参数

```bash
torchrun --master_port 29503 pretrain.py --config ...
```

### 数据加载错误

**解决方案**：验证配置文件中的数据路径和键名

### 性能优化建议

1. 使用多GPU训练
2. 启用AMP（自动混合精度）
3. 优化数据加载（增加DataLoader的num_workers）
4. 预处理数据一次后重复使用
5. 使用flash attention

## 项目结构

```
EpiFoundation_dev/
├── pretrain.py          # 预训练脚本
├── finetune.py          # 微调脚本
├── eval.py              # 评估脚本
├── prepare_data.py      # 数据准备工具
├── model/               # 模型架构
├── data/                # 数据处理
├── tokenizer/           # 词表管理
├── loss/                # 损失函数
├── configs/             # 配置文件
├── scripts/             # Shell脚本
├── docs/                # 详细文档
└── utils.py             # 工具函数
```

## 文档说明

- **README.md**：英文完整文档
- **README_CN.md**：中文文档（本文件）
- **QUICKSTART.md**：5分钟快速开始指南
- **docs/DATA_PREPARATION.md**：数据准备详细指南
- **docs/CONFIGURATION_GUIDE.md**：配置参数完整参考

## 引用

如果您在研究中使用本代码，请引用：

```bibtex
@article{epifoundation2024,
  title={EpiFoundation: A Foundation Model for Single-Cell Multi-Omics},
  author={Your Name},
  journal={bioRxiv},
  year={2024}
}
```

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题或建议，请在GitHub上提issue。

---

**祝您训练顺利！🧬🔬**

