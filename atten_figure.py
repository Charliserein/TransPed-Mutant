import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from copy import deepcopy
import re
import os

# 用于处理文件名
def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

# 排序氨基酸类型
def sort_aatype(df):
    aatype_sorts = list('ARNDCQEGHILKMFPSTWYV')
    df.reset_index(inplace=True)
    df['index'] = df['index'].astype('category')
    df['index'].cat.reorder_categories(aatype_sorts, inplace=True)
    df.sort_values('index', inplace=True)
    df.rename(columns={'index': ''}, inplace=True)
    df = df.set_index('')
    return df

# 计算注意力权重：基于peptide A 和 peptide B
def attn_sumhead_pepA_pepB(data, attn_data, label=None):
    SUM_pepA_pepB_dict = {}
    for l in range(8, 15):  # 假设peptide A 和 B 的长度都在8到14之间
        print('Length = ', str(l))
        SUM_pepA_pepB_dict[l] = []
        
        if label is None:
            length_index = np.array(data[data.length == l].index)
        elif label == 1:
            length_index = np.array(data[data.label == 1][data.length == l].index)
        elif label == 0:
            length_index = np.array(data[data.label == 0][data.length == l].index)
            
        length_data_num = len(length_index)
        print(length_data_num, length_index)

        for head in range(9):
            idx_0 = length_index[0]
            temp_length_head = deepcopy(nn.Softmax(dim=-1)(attn_data[idx_0][head][:, :l].float()))  # 注意力权重初始化

            for idx in length_index[1:]:
                temp_length_head += nn.Softmax(dim=-1)(attn_data[idx][head][:, :l].float())

            temp_length_head = np.array(nn.Softmax(dim=-1)(temp_length_head.sum(axis=0)))  # 聚合注意力权重
            SUM_pepA_pepB_dict[l].append(temp_length_head)
            
    #############################
    SUM_pepA_pepB_sum = []
    for l in range(8, 15):
        temp = pd.DataFrame(SUM_pepA_pepB_dict[l], columns=range(1, l+1)).round(4)
        temp.loc['sum'] = temp.sum(axis=0)
        SUM_pepA_pepB_sum.append(list(temp.loc['sum']))
        print(l, temp.loc['sum'].sort_values(ascending=False).index)
        
    return SUM_pepA_pepB_dict, SUM_pepA_pepB_sum

# 绘制不同长度的pep A 和 pep B的注意力分布
def draw_eachlength_pepA_pepB(sum_pepA_pepB, label, savepath=False):
    
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 6), dpi=600)
    fig.patch.set_facecolor('white')
    cmap = 'cividis'
    
    for l in range(8, 15):
        temp = pd.DataFrame(sum_pepA_pepB[l], columns=range(1, l+1)).round(4)
        sns.heatmap(temp, cmap=cmap, cbar=False, square=True, ax=axes[(l-8)//4, (l-8)%4],
                    xticklabels=range(1, l+1), yticklabels=range(1, 10))
        axes[(l-8)//4, (l-8)%4].set_title('Peptide A-B Length = {}'.format(l))
        
    axes[0, 0].set_ylabel('Head')
    axes[1, 0].set_ylabel('Head')
    axes[0, 3].set_xlabel('Peptide position')
    
    label = 'All' if label is None else ['Negative', 'Positive'][label == 1]
    fig.suptitle('{} samples | Heads - Peptide positions'.format(label), x=0.43)
    
    if savepath:
        plt.savefig('./figures/Attention/{} samples_eachPepA_PepB_Length.tif'.format(label), dpi=600, bbox_inches='tight')

    plt.show()

# 针对每一个head：绘制pep A 和 pep B 的位置映射
def draw_eachhead_pepA_pepB(SUM_pepA_pepB_dict, label=None, savepath=False):
    
    SUM_head_pepA_pepB_dict = dict()
    for l in range(8, 15):
        for head in range(9):
            SUM_head_pepA_pepB_dict.setdefault(head, [])
            SUM_head_pepA_pepB_dict[head].append(SUM_pepA_pepB_dict[l][head])
    assert len(SUM_head_pepA_pepB_dict[1]) == 7
    assert len(SUM_head_pepA_pepB_dict.keys()) == 9

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 7), dpi=600)
    fig.patch.set_facecolor('white')
    cmap = 'cividis'
    
    for head in range(9):
        temp = pd.DataFrame(SUM_head_pepA_pepB_dict[head], columns=range(1, 15), index=range(8, 15))
        sns.heatmap(temp, cmap=cmap, cbar=False, square=True, xticklabels=True, yticklabels=True, ax=axes[head//3, head%3])
        axes[head//3, head%3].set_title('Head '+ str(head))
        
    axes[0, 0].set_ylabel('Peptide A-B length')
    axes[1, 0].set_ylabel('Peptide A-B length')
    axes[2, 0].set_xlabel('Peptide position')
    
    label = 'All' if label is None else ['Negative', 'Positive'][label == 1]
    fig.suptitle('{} samples | Peptide A-B lengths - Positions'.format(label), x=0.42)
    
    if savepath:
        plt.savefig('./figures/Attention/{} samples_eachHead_PepA_PepB_Positions.tif'.format(label), dpi=600, bbox_inches='tight')

    plt.show()

# 对 pepA 和 pepB 的氨基酸类型与位置的可视化
def attn_pepA_pepB_aatype_position_num(data, attn_data, pepA='SISELVAYL', pepB='KAGLYSD', label=None, length=9, show_num=False):
    aatype_position = dict()
    if label is None:
        length_index = np.array(data[data.length == length][data.pepA == pepA][data.pepB == pepB].index)
    else:
        length_index = np.array(data[data.length == length][data.pepA == pepA][data.pepB == pepB][data.label == label].index)

    length_data_num = len(length_index)

    for head in range(9):
        for idx in length_index:
            temp_peptide_A = data.iloc[idx].pepA
            temp_peptide_B = data.iloc[idx].pepB
            temp_length_head = deepcopy(nn.Softmax(dim=-1)(attn_data[idx][head][:, :length].float())).sum(axis=0)
            
            for i, aa in enumerate(temp_peptide_A): 
                aatype_position.setdefault(aa, {})
                aatype_position[aa].setdefault(i, 0)
                aatype_position[aa][i] += temp_length_head[i]
                
    if show_num:
        aatype_position_num = dict()
        for idx in length_index:
            temp_peptide_A = data.iloc[idx].pepA
            temp_peptide_B = data.iloc[idx].pepB
            for i, aa in enumerate(temp_peptide_A):
                aatype_position_num.setdefault(aa, {})
                aatype_position_num[aa].setdefault(i, 0)
                aatype_position_num[aa][i] += 1
             
        return aatype_position, aatype_position_num
    else:
        return aatype_position

# 可视化pep A 和 pep B氨基酸类型-位置的注意力权重
def draw_pepA_pepB_aatype_position(data, attn_data, pepA='SISELVAYL', pepB='KAGLYSD', label=None, length=9, 
                                    show=True, softmax=True, unsoftmax=True):
    
    pepA_pepB_aatype_position = attn_pepA_pepB_aatype_position_num(data, attn_data, pepA, pepB, label, length, show_num=False)
    
    if softmax and unsoftmax:
        pepA_pepB_aatype_position_softmax_pd, pepA_pepB_aatype_position_unsoftmax_pd = attn_HLA_length_aatype_position_pd(
                                                                                     pepA_pepB_aatype_position, 
                                                                                     length, 
                                                                                     softmax,
                                                                                     unsoftmax)
        pepA_pepB_aatype_position_softmax_pd = sort_aatype(pepA_pepB_aatype_position_softmax_pd)
        pepA_pepB_aatype_position_unsoftmax_pd = sort_aatype(pepA_pepB_aatype_position_unsoftmax_pd)
        
        if show:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))
            sns.heatmap(pepA_pepB_aatype_position_softmax_pd,
                        ax=axes[0], cmap='YlGn', square=True)

            sns.heatmap(pepA_pepB_aatype_position_unsoftmax_pd,
                        ax=axes[1], cmap='YlGn', square=True)

            axes[0].set_title('Pep A-B Softmax Normalization')
            axes[1].set_title('Pep A-B UnNormalization')
            plt.show()

        return pepA_pepB_aatype_position_softmax_pd, pepA_pepB_aatype_position_unsoftmax_pd
    
    else:
        pepA_pepB_aatype_position_pd = attn_HLA_length_aatype_position_pd(pepA_pepB_aatype_position, 
                                                                           length, 
                                                                           softmax,
                                                                           unsoftmax)
        pepA_pepB_aatype_position_pd = sort_aatype(pepA_pepB_aatype_position_pd)
        return pepA_pepB_aatype_position_pd

