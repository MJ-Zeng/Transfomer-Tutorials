"""
Transformer 教程辅助工具模块
包含画图函数等辅助功能，让初学者不必看到复杂的画图代码
"""

import matplotlib
# Force matplotlib to find system Chinese fonts
import matplotlib.font_manager as fm
for fp in [r'C:\Windows\Fonts\simhei.ttf', r'C:\Windows\Fonts\msyh.ttc']:
    try:
        fm.fontManager.addfont(fp)
    except:
        pass
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_attention_heatmap(attention_weights, x_label='Keys', y_label='Queries',
                            title=None, figsize=(10, 6)):
    """
    画注意力权重热力图
    
    参数:
        attention_weights: 形状 (batch, heads, query_len, key_len) 或 (batch, query_len, key_len)
        x_label: x轴标签
        y_label: y轴标签
        title: 子图标题列表
        figsize: 图大小
    """
    # 确保是4D: (batch, heads, query_len, key_len)
    if attention_weights.dim() == 3:
        attention_weights = attention_weights.unsqueeze(1)
    
    batch_size, n_heads, q_len, k_len = attention_weights.shape
    weights = attention_weights.detach().cpu().numpy()
    
    total_plots = batch_size * n_heads
    cols = min(total_plots, 4)
    rows = (total_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0], figsize[1] * rows))
    axes = axes.flatten() if total_plots > 1 else [axes]
    
    for i in range(total_plots):
        if i >= len(axes):
            break
        batch_idx = i // n_heads
        head_idx = i % n_heads
        
        im = axes[i].imshow(weights[batch_idx, head_idx], cmap='Blues', aspect='auto')
        axes[i].set_xlabel(x_label, fontsize=10)
        axes[i].set_ylabel(y_label, fontsize=10)
        
        if title and i < len(title):
            axes[i].set_title(title[i], fontsize=9)
        
        # 在格子里显示数值
        for qi in range(q_len):
            for ki in range(k_len):
                val = weights[batch_idx, head_idx, qi, ki]
                axes[i].text(ki, qi, f'{val:.2f}', ha='center', va='center',
                            fontsize=7, color='black' if val < 0.7 else 'white')
        
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    # 隐藏多余的子图
    for i in range(total_plots, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_positional_encoding(pos_encoding, max_len=100, d_model=512):
    """
    画位置编码热力图
    
    参数:
        pos_encoding: 位置编码矩阵，形状 (max_len, d_model)
        max_len: 序列最大长度
        d_model: 模型维度
    """
    pe = pos_encoding.detach().cpu().numpy() if torch.is_tensor(pos_encoding) else pos_encoding
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # 热力图
    cax = axes[0].imshow(pe[:max_len, :d_model], aspect='auto', cmap='RdBu')
    axes[0].set_xlabel('维度索引')
    axes[0].set_ylabel('位置索引')
    axes[0].set_title(f'位置编码热力图 (max_len={max_len}, d_model={d_model})')
    fig.colorbar(cax, ax=axes[0])
    
    # 展示前几个维度的曲线
    for i in range(min(4, d_model)):
        axes[1].plot(range(max_len), pe[:max_len, i], label=f'维度 {i}')
    axes[1].set_xlabel('位置')
    axes[1].set_ylabel('编码值')
    axes[1].set_title(f'位置编码曲线 (前{min(4, d_model)}个维度)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'总参数量: {total:,}')
    print(f'可训练参数量: {trainable:,}')
    return total, trainable