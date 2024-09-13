from pre_train import *
from mutant import *
from atten_figure import transformer_attention_weights
import os
import argparse
import sys


import numpy as np
import pandas as pd
import re
import random
import torch

# 设置随机种子确保实验可复现
seed = 13326543
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def sanitize_filename(filename):
    """清理文件名中的特殊字符"""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

# 设置参数解析器
parser = argparse.ArgumentParser(usage='Pep A 和 Pep B结合预测')
parser.add_argument('--pepA_file', type=str, help='包含肽A序列的.fasta文件路径')
parser.add_argument('--pepB_file', type=str, help='包含肽B序列的.fasta文件路径')
parser.add_argument('--threshold', type=float, default=0.5, help='定义预测结合的阈值，范围为0-1')
parser.add_argument('--cut_peptide', type=bool, default=True, help='是否分割长度大于cut_length的肽')
parser.add_argument('--cut_length', type=int, default=9, help='分割肽的最大长度，默认为9')
parser.add_argument('--output_dir', type=str, help='输出结果保存的目录')
parser.add_argument('--output_attention', type=bool, default=True, help='是否输出Pep A和Pep B的注意力权重')
parser.add_argument('--output_heatmap', type=bool, default=True, help='是否可视化Pep A和Pep B的结合影响')
parser.add_argument('--output_mutation', type=bool, default=True, help='是否为每个样本进行优化突变以提高结合亲和力')

args = parser.parse_args()

# 错误日志路径
errLogPath = os.path.join(args.output_dir, 'error.log')
if args.threshold <= 0 or args.threshold >= 1:
    log = Logger(errLogPath)
    log.logger.critical('无效阈值，请确保其在0-1范围内')
    sys.exit(0)

if not args.pepA_file:
    log = Logger(errLogPath)
    log.logger.critical('肽A文件为空')
    sys.exit(0)
if not args.pepB_file:
    log = Logger(errLogPath)
    log.logger.critical('肽B文件为空')
    sys.exit(0)
if not args.output_dir:
    log = Logger(errLogPath)
    log.logger.critical('请填写输出文件目录')
    sys.exit(0)

# 创建输出目录
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

cut_length = args.cut_length

# 读取肽A和肽B序列文件
with open(args.pepA_file, 'r') as f:
    pepA_file = f.readlines()

with open(args.pepB_file, 'r') as f:
    pepB_file = f.readlines()

if len(pepA_file) != len(pepB_file):
    log = Logger(errLogPath)
    log.logger.critical('请确保肽A和肽B的数量相同')
    sys.exit(0)

# 处理序列
ori_pepA, ori_pepB = [], []
for i, (pepA, pepB) in enumerate(zip(pepA_file, pepB_file)):
    if i % 2 == 1:
        pepA_seq = str.upper(pepA.replace('\n', '').replace('\t', ''))
        pepB_seq = str.upper(pepB.replace('\n', '').replace('\t', ''))
        ori_pepA.append(pepA_seq)
        ori_pepB.append(pepB_seq)

peptides_A, peptides_B = [], []
for pepA, pepB in zip(ori_pepA, ori_pepB):
    # 检查序列是否仅包含合法氨基酸字符
    if not (pepA.isalpha() and pepB.isalpha()): 
        continue
    if len(set(pepA).difference(set('ARNDCQEGHILKMFPSTWYV'))) != 0:
        continue
    if len(set(pepB).difference(set('ARNDCQEGHILKMFPSTWYV'))) != 0:
        continue

    # 肽的长度分割逻辑
    if len(pepA) < 15 and len(pepB) < 15:
        if args.cut_peptide:
            if len(pepA) > cut_length:
                cut_pepA = [pepA[i:i + cut_length] for i in range(len(pepA) - cut_length + 1)]
                cut_pepB = [pepB[i:i + cut_length] for i in range(len(pepB) - cut_length + 1)]
                peptides_A.extend(cut_pepA)
                peptides_B.extend(cut_pepB)
            else:
                peptides_A.append(pepA)
                peptides_B.append(pepB)
        else:
            peptides_A.append(pepA)
            peptides_B.append(pepB)
            
# 创建预测数据表
predict_data = pd.DataFrame([peptides_A, peptides_B], index=['pepA', 'pepB']).T
if predict_data.shape[0] == 0:
    log = Logger(errLogPath)
    log.logger.critical('没有可预测的数据，请检查输入')
    sys.exit(0)

# 准备数据进行预测
predict_data, predict_loader = read_predict_data(predict_data, batch_size)

# 模型预测
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model_file = 'Transformer_pepAB.pkl'  # 保存的Transformer模型文件

model_eval = Transformer().to(device)
model_eval.load_state_dict(torch.load(model_file, map_location=device), strict=True)

model_eval.eval()
y_pred, y_prob, attns = eval_step(model_eval, predict_loader, args.threshold, use_cuda)

# 保存预测结果
predict_data['y_pred'], predict_data['y_prob'] = y_pred, y_prob
predict_data = predict_data.round({'y_prob': 4})
predict_data.to_csv(args.output_dir + '/predict_results.csv', index=False)

# 输出注意力权重和热力图
if args.output_attention or args.output_heatmap:
    if args.output_attention:
        attn_savepath = os.path.join(args.output_dir, 'attention/')
        if not os.path.exists(attn_savepath):
            os.makedirs(attn_savepath)
    else:
        attn_savepath = False

    if args.output_heatmap:
        fig_savepath = os.path.join(args.output_dir, 'figures/')
        if not os.path.exists(fig_savepath):
            os.makedirs(fig_savepath)
    else:
        fig_savepath = False

    for pepA, pepB in zip(predict_data.pepA, predict_data.pepB):
        transformer_attention_weights(predict_data, attns, pepA, pepB, attn_savepath, fig_savepath)

# 突变优化
if args.output_mutation:
    mut_savepath = os.path.join(args.output_dir, 'mutation/')
    if not os.path.exists(mut_savepath):
        os.makedirs(mut_savepath)

    for idx in range(predict_data.shape[0]):
        pepA = predict_data.iloc[idx].pepA
        pepB = predict_data.iloc[idx].pepB

        # 生成突变肽并进行重新预测
        mut_peptides_df = pHLA_mutation_peptides(predict_data, attns, pepA=pepA, pepB=pepB)
        mut_data, mut_loader = read_predict_data(mut_peptides_df, batch_size)

        model_eval.eval()
        y_pred, y_prob, attns = eval_step(model_eval, mut_loader, args.threshold, use_cuda)

        mut_data['y_pred'], mut_data['y_prob'] = y_pred, y_prob
        mut_data = mut_data.round({'y_prob': 4})
        
        # 保存突变结果
        sanitized_pepA = sanitize_filename(pepA)
        sanitized_pepB = sanitize_filename(pepB)
        file_path = os.path.join(mut_savepath, f'{sanitized_pepA}_{sanitized_pepB}_mutation.csv')
        mut_data.to_csv(file_path, index=False)
        print(f'********** {pepA} | {pepB} → 突变肽数量 = {mut_data.shape[0] - 1}')
