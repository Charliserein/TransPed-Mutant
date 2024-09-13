import numpy as np
import pandas as pd
import difflib
from collections import OrderedDict
from attention import transformer_attention_weights  # 假设这里是Transformer的attention权重提取函数

def pep_attention_keyaatype_keyaacontrib(data, attns, pep_a=False, pep_b=False): 
    # 提取 Pep A 和 Pep B 交互的attention权重数据
    pep_attns_pd, pep_a, pep_b = transformer_attention_weights(data, attns, pep_a, pep_b)

    # 提取Attention中的氨基酸类型及其贡献
    pep_keyaatype = OrderedDict()
    for posi, pep_aa, aa_attn_sum in zip(list(pep_attns_pd.loc['posi']),
                                         list(pep_attns_pd.columns),
                                         list(pep_attns_pd.loc['sum'])):
        pep_keyaatype[int(posi)] = [pep_aa, aa_attn_sum]
    pep_keyaatype = OrderedDict(sorted(pep_keyaatype.items(), key=lambda t: (-t[1][1])))

    pep_keyaatype_contrib = OrderedDict()
    for posi, pep_aa, aa_attn_contrib in zip(list(pep_attns_pd.loc['posi']),
                                             list(pep_attns_pd.columns),
                                             list(pep_attns_pd.loc['contrib'])):
        pep_keyaatype_contrib[int(posi)] = [pep_aa, aa_attn_contrib]
    pep_keyaatype_contrib = OrderedDict(sorted(pep_keyaatype_contrib.items(), key=lambda t: (-t[1][1])))

    return pep_attns_pd, pep_keyaatype, pep_keyaatype_contrib

def pep_length_aatype_position_num(pep_a=False, length=9, label=None):
    # 根据长度和标签加载氨基酸位点数据
    if label == 'None': 
        new_label = 'all'
    elif label == 1:
        new_label = 'positive'
    elif label == 0:
        new_label = 'negative'
    
    try:
        aatype_position = np.load(f'./Attention/pepAAtype_pepPosition/{pep_a}_Length{length}.npy', allow_pickle=True).item()[new_label]
        aatype_position_num = np.load(f'./Attention/pepAAtype_pepPosition_NUM/{pep_a}_Length{length}_num.npy', allow_pickle=True).item()[new_label]
    except:
        print(f'No {pep_a} with {length}, Use the overall attention for pepAAtype-peppsition')
        aatype_position = np.load('./Attention/pepAAtype_pepPosition/Allsamples_Alllengths.npy', allow_pickle=True).item()[length][new_label]
        aatype_position_num = np.load('./Attention/pepAAtype_pepPosition_NUM/Allsamples_Alllengths_num.npy', allow_pickle=True).item()[length][new_label]
    
    aatype_position.loc['sum'] = aatype_position.sum(axis=0)
    aatype_position['sum'] = aatype_position.sum(axis=1)
    
    return aatype_position, aatype_position_num

def pep_aatype_position_contribution(aatype_position_pd, aatype_position_num_pd, length=9):
    contrib = np.zeros((20, length))
    for aai, aa in enumerate(aatype_position_pd.index[:-1]):  # 处理sum行
        for pi, posi in enumerate(aatype_position_pd.columns[:-1]):  # 处理sum列
            p_aa_posi = aatype_position_pd.loc[aa, posi] / aatype_position_num_pd.loc[aa, posi]
            p_posi = aatype_position_num_pd.loc['sum', 'sum'] / aatype_position_pd.loc['sum', 'sum']
            contrib[aai, pi] = p_aa_posi * p_posi
            
    contrib = pd.DataFrame(contrib, index=aatype_position_pd.index[:-1], columns=aatype_position_pd.columns[:-1])
    contrib.fillna(0, inplace=True)
    return contrib

def pep_length_label_pepaatype_pepposition(pep_a=False, length=9, label=1):
    aatype_position_pd, aatype_position_num_pd = pep_length_aatype_position_num(pep_a, length, label)
    aatype_position_contrib_pd = pep_aatype_position_contribution(aatype_position_pd, aatype_position_num_pd, length)
    return aatype_position_contrib_pd, aatype_position_pd, aatype_position_num_pd

def pep_length_position_keyaatype(aatype_position_pd, aatype_position_contrib_pd, length=9):
    # 生成氨基酸及位点的贡献排序
    position_contrib_keyaatype = OrderedDict()
    for posi in range(1, length + 1):
        temp_sorted = aatype_position_contrib_pd[posi].sort_values(ascending=False)
        key_aatype = [k for k, v in OrderedDict(temp_sorted > 1).items() if v]
        position_contrib_keyaatype[posi] = [key_aatype, len(key_aatype), temp_sorted.max().round(2), temp_sorted.mean().round(2)]
    position_contrib_keyaatype = OrderedDict(sorted(position_contrib_keyaatype.items(), key=lambda t: (-t[1][2], t[1][1], -t[1][3])))
    
    aatype_position_pd = aatype_position_pd.drop(index='sum')
    aatype_position_pd = aatype_position_pd.drop(columns='sum')
    position_keyaatype = OrderedDict()
    for posi in range(1, length + 1):
        temp_sorted = aatype_position_pd[posi].sort_values(ascending=False)
        key_aatype = [k for k, v in OrderedDict(temp_sorted > aatype_position_pd[posi].mean()).items() if v]
        position_keyaatype[posi] = [key_aatype, len(key_aatype), temp_sorted.max().round(2), temp_sorted.mean().round(2)]
    position_keyaatype = OrderedDict(sorted(position_keyaatype.items(), key=lambda t: (t[1][1], -t[1][2], t[1][3])))
    
    return position_contrib_keyaatype, position_keyaatype

def pep_length_aatype_position_contrib_attn_num(pep_a=False, length=9, label=1):
    aatype_position_contrib_pd, aatype_position_pd, aatype_position_num_pd = pep_length_label_pepaatype_pepposition(pep_a, length, label)
    position_contrib_keyaatype, position_keyaatype = pep_length_position_keyaatype(aatype_position_pd, aatype_position_contrib_pd, length)
    return position_contrib_keyaatype, position_keyaatype, aatype_position_contrib_pd, aatype_position_pd, aatype_position_num_pd

def pep_interaction_contrib_keyaatype_attn_num(data, attn_data, pep_a=False, pep_b=False, label=1):
    length = len(pep_b)

    # 正样本预测
    position_contrib_keyaatype, position_keyaatype, aatype_position_contrib_pd, aatype_position_pd, aatype_position_num_pd = pep_length_aatype_position_contrib_attn_num(pep_a, length, label)

    # 提取Pep A和Pep B的attention
    pep_attns_pd, pep_keyaatype, pep_keyaatype_contrib = pep_attention_keyaatype_keyaacontrib(data, attn_data, pep_a, pep_b)
    
    return position_contrib_keyaatype, position_keyaatype, aatype_position_contrib_pd, aatype_position_pd, aatype_position_num_pd, pep_attns_pd, pep_keyaatype, pep_keyaatype_contrib

def oneposition_mut_peptides(mut_posi, mut_aatypes, peptide, print_=False):
    mut_peptides = []
    for mut_aa in mut_aatypes:
        index = mut_posi - 1
        mut_peptide = peptide[:index] + mut_aa + peptide[index + 1:]
        mut_peptides.append(mut_peptide)
    if print_: print(mut_peptides)
    return mut_peptides

def find_mutate_position_aatype(pep_a_oripeptide_all_mutpeptides):
    original_peptide = pep_a_oripeptide_all_mutpeptides[0]
    pep_length = len(original_peptide)
    mutate_position_aatype, mutate_num = [], []
    for pep in pep_a_oripeptide_all_mutpeptides:
        s = ''
        for i in range(pep_length):
            if original_peptide[i] != pep[i]:
                s += f"{i + 1}|{original_peptide[i]}/{pep[i]},"
        mutate_num.append(len(s[:-1].split(',')))
        mutate_position_aatype.append(s[:-1])
    return mutate_position_aatype, mutate_num

def mutation_pep_strategy_1(pep_keyaatype_contrib, pep_length_position_contrib_keyaatype, pep_length_position_keyaatype, peptide='PEPTIDEA', pep_a='PepA', print_=False):
    # 策略1：使用贡献值排序突变位点
    mut_peptides_step = dict()
    mut_peptides_step[0] = peptide
    all_peptides = []
    mut_positions = []

    i = 0
    while i <= 3:
        mut_posi = sorted(pep_keyaatype_contrib.items(), key=lambda x: x[1][1], reverse=True)[i][0]
        mut_aatypes = pep_length_position_keyaatype[mut_posi][0][:3]  # 挑选前3个突变氨基酸
        if print_: print(f"Mutate position {mut_posi} in peptide {peptide} with aatypes {mut_aatypes}")
        mut_peptides = oneposition_mut_peptides(mut_posi, mut_aatypes, peptide)
        mut_peptides_step[i+1] = mut_peptides
        all_peptides.extend(mut_peptides)
        mut_positions.append(mut_posi)
        i += 1
    
    return mut_peptides_step, all_peptides

def mutation_pep_strategy_2(pep_keyaatype, pep_length_position_keyaatype, peptide='PEPTIDEA', pep_a='PepA', print_=False):
    # 策略2：优先考虑正样本贡献
    mut_peptides_step = dict()
    mut_peptides_step[0] = peptide
    all_peptides = []
    mut_positions = []

    i = 0
    while i <= 3:
        mut_posi = sorted(pep_keyaatype.items(), key=lambda x: x[1][1], reverse=True)[i][0]
        mut_aatypes = pep_length_position_keyaatype[mut_posi][0][:3]
        if print_: print(f"Mutate position {mut_posi} in peptide {peptide} with aatypes {mut_aatypes}")
        mut_peptides = oneposition_mut_peptides(mut_posi, mut_aatypes, peptide)
        mut_peptides_step[i+1] = mut_peptides
        all_peptides.extend(mut_peptides)
        mut_positions.append(mut_posi)
        i += 1
    
    return mut_peptides_step, all_peptides

def mutation_pep_strategy_3(pep_keyaatype, pep_length_position_keyaatype, peptide='PEPTIDEA', pep_a='PepA', print_=False):
    # 策略3：优先考虑负样本贡献
    mut_peptides_step = dict()
    mut_peptides_step[0] = peptide
    all_peptides = []
    mut_positions = []

    i = 0
    while i <= 3:
        mut_posi = sorted(pep_keyaatype.items(), key=lambda x: x[1][1], reverse=False)[i][0]
        mut_aatypes = pep_length_position_keyaatype[mut_posi][0][:3]
        if print_: print(f"Mutate position {mut_posi} in peptide {peptide} with aatypes {mut_aatypes}")
        mut_peptides = oneposition_mut_peptides(mut_posi, mut_aatypes, peptide)
        mut_peptides_step[i+1] = mut_peptides
        all_peptides.extend(mut_peptides)
        mut_positions.append(mut_posi)
        i += 1
    
    return mut_peptides_step, all_peptides

def mutation_pep_strategy_4(pep_keyaatype, pep_length_position_keyaatype, peptide='PEPTIDEA', pep_a='PepA', print_=False):
    # 策略4：通过固定位置组合突变
    mut_peptides_step = dict()
    mut_peptides_step[0] = peptide
    all_peptides = []
    mut_positions = []

    i = 0
    while len(mut_positions) < 4:
        mut_posi = sorted(pep_keyaatype.items(), key=lambda x: x[1][1], reverse=True)[i][0]
        mut_aatypes = pep_length_position_keyaatype[mut_posi][0][:3]
        if print_: print(f"Mutate position {mut_posi} in peptide {peptide} with aatypes {mut_aatypes}")
        mut_peptides = oneposition_mut_peptides(mut_posi, mut_aatypes, peptide)
        mut_peptides_step[i+1] = mut_peptides
        all_peptides.extend(mut_peptides)
        mut_positions.append(mut_posi)
        i += 1
    
    return mut_peptides_step, all_peptides

def pep_mutation_peptides(data, attn_data, idx=-1, pep_a=False, pep_b=False, print_=False):
    if not(pep_b and pep_a) and idx > -1:
        pep_a = data.iloc[idx].PepA
        pep_b = data.iloc[idx].PepB
    
    pep_length_position_contrib_keyaatype, pep_length_position_keyaatype, pep_length_aatype_position_contrib_pd, pep_length_aatype_position_pd, pep_length_aatype_position_num_pd, pep_attns_pd, pep_keyaatype, pep_keyaatype_contrib = pep_interaction_contrib_keyaatype_attn_num(data, attn_data, pep_a, pep_b, label=1)

    mut_peptides = []
    if print_: print('********** Strategy 1 **********')
    pep_mut_peptides_step_1, pep_mut_peptides_1 = mutation_pep_strategy_1(pep_keyaatype_contrib, pep_length_position_contrib_keyaatype, pep_length_position_keyaatype, peptide=pep_b, pep_a=pep_a, print_=print_)
    if print_: print('********** Strategy 2 **********')
    pep_mut_peptides_step_2, pep_mut_peptides_2 = mutation_pep_strategy_2(pep_keyaatype, pep_length_position_keyaatype, peptide=pep_b, pep_a=pep_a, print_=print_)
    if print_: print('********** Strategy 3 **********')
    pep_mut_peptides_step_3, pep_mut_peptides_3 = mutation_pep_strategy_3(pep_keyaatype, pep_length_position_keyaatype, peptide=pep_b, pep_a=pep_a, print_=print_)
    if print_: print('********** Strategy 4 **********')
    pep_mut_peptides_step_4, pep_mut_peptides_4 = mutation_pep_strategy_4(pep_keyaatype, pep_length_position_keyaatype, peptide=pep_b, pep_a=pep_a, print_=print_)

    mut_peptides.extend(pep_mut_peptides_1)
    mut_peptides.extend(pep_mut_peptides_2)
    mut_peptides.extend(pep_mut_peptides_3)
    mut_peptides.extend(pep_mut_peptides_4)

    mut_peptides = list(set(mut_peptides))
    mut_peptides = [pep_b] + mut_peptides

    mutate_position_aatype = find_mutate_position_aatype(mut_peptides)

    all_peptides_df = pd.DataFrame([[pep_b] * len(mut_peptides),
                                    mut_peptides,
                                    mutate_position_aatype[0],
                                    mutate_position_aatype[1],
                                    [difflib.SequenceMatcher(None, item, pep_b).ratio() for item in mut_peptides]],
                                   index=['original_peptide', 'mutation_peptide', 'mutation_position_AAtype', 'mutation_AA_number', 'sequence similarity']).T.drop_duplicates().sort_values(by='mutation_AA_number').reset_index(drop=True)
    all_peptides_df['PepA'] = pep_a
    return all_peptides_df  # 返回所有突变肽序列

