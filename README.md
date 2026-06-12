# Transformer 从零实现教程

本教程从零开始，使用 PyTorch 逐步实现完整的 Transformer 模型，包含 5 个循序渐进的 Jupyter Notebook。无论你是深度学习初学者还是有经验的开发者，都可以通过本教程深入理解 Transformer 的原理与实现。

## 教程目录

| 章节      | 文件                           | 内容                                                           |
| --------- | ------------------------------ | -------------------------------------------------------------- |
| **第1章** | `01_attention_mechanism.ipynb` | 注意力机制：从 QKV 框架到多头注意力、自注意力                  |
| **第2章** | `02_positional_encoding.ipynb` | 位置编码：置换等变性、Sinusoidal 编码                          |
| **第3章** | `03_transformer_model.ipynb`   | Transformer 完整模型：Encoder-Decoder、学习率调度、Beam Search |
| **第4章** | `04_bpe_tokenization.ipynb`    | BPE 子词分词：Byte Pair Encoding 完整实现                      |
| **第5章** | `05_data_preprocessing.ipynb`  | 数据预处理：多语言平行语料处理流水线                           |

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.10+
- Jupyter Notebook / Jupyter Lab

### 安装依赖

```bash
pip install torch torchtext numpy matplotlib seaborn spacy tqdm dill
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

### 运行教程

按顺序打开 Jupyter Notebook：

```bash
jupyter notebook 01_attention_mechanism.ipynb
```

建议按章节编号顺序学习，每章都包含完整的数学推导、代码实现和可视化分析。

## 项目结构

```
├── 01_attention_mechanism.ipynb   # 第1章：注意力机制
├── 02_positional_encoding.ipynb   # 第2章：位置编码
├── 03_transformer_model.ipynb     # 第3章：Transformer 模型架构
├── 04_bpe_tokenization.ipynb      # 第4章：BPE 子词分词
├── 05_data_preprocessing.ipynb    # 第5章：数据预处理
├── img/                           # 教程图片资源
│   ├── 1_multi-head.png           # 多头注意力示意图
│   ├── 2_self_attention.png       # 自注意力示意图
│   ├── 3_1_translator.png         # 翻译示意图
│   ├── 3_2_beam_search.png        # Beam Search 流程
│   ├── 3_3_BeamSearchVisua1.png   # Beam Search 可视化1
│   ├── 3_4_BeamSearchVisua2.png   # Beam Search 可视化2
│   ├── 3_5_BeamSearchVisua3.png   # Beam Search 可视化3
│   └── 4_1_zipfs-law.png          # Zipf 定律图
├── multi30k_de_en.pkl             # 预处理后的 Multi30k 数据集
├── tutorial_prompt.md             # 教程写作参考文档
└── README.md                      # 本文件
```

## 各章内容概览

### 第1章：注意力机制
- QKV 数学框架（"图书馆找书"类比）
- 5 种注意力评分函数（点积、缩放点积、加性、乘性、多头）
- Nadaraya-Watson 核回归（带宽与注意力关系）
- 掩蔽 Softmax、加性/缩放点积/多头注意力代码实现
- 自注意力机制完整数学推导与代码实现

### 第2章：位置编码
- 置换等变性数学证明
- Sinusoidal 位置编码公式与频率特性分析
- 代码实现与热力图可视化

### 第3章：Transformer 模型架构
- Encoder-Decoder 完整架构
- Multi-Head Attention + Positionwise FFN + Positional Encoding
- 填充掩码与因果掩码
- 权重共享策略
- 学习率调度（Warmup + Decay）
- Beam Search 解码策略

### 第4章：BPE 子词分词
- OOV 问题分析与传统分词方案对比
- BPE 核心原理与 Zipf 定律
- 完整 BPE 学习、编码、解码代码实现

### 第5章：数据预处理
- 多语言平行语料处理流水线
- 下载、解压、合并、BPE 编码
- 词汇表构建与数据集保存
- 双模式支持（BPE 模式 / spacy 模式）

## 许可

MIT License