
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from copy import deepcopy
import re
import numpy as np
import pandas as pd
import random
import warnings
import math
from sklearn import metrics
import os


# 忽略不必要的警告
warnings.filterwarnings("ignore")

# 清理文件名中的特殊字符
def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


# 设置随机种子以确保实验可复现
seed = 13326543
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



# 设置肽链A和肽链B的最大长度
pep_a_max_len = 11 
pep_b_max_len = 11 
tgt_len = pep_a_max_len + pep_b_max_len  # 目标长度为两个肽链的长度总和

# 加载词典
vocab = np.load('test_dict.npy', allow_pickle=True).item()
vocab_size = len(vocab)


# Transformer 参数
d_model = 64  # Embedding 尺寸
d_ff = 512  # 前馈网络隐藏层尺寸
d_k = d_v = 64  # Q, K, V的维度
d_model = 64  # Embedding大小
d_ff = 512  # FeedForward层的维度
d_k = d_v = 64  # K, V的维度
n_layers = 1  # 编码器和解码器的层数
n_heads = 9  # 多头注意力机制中的头数
batch_size = 1024  # 批大小


# 读取预测数据，并生成肽链A和肽链B的向量
def load_prediction_data(predict_data, batch_size):
    pep_sequence = pd.read_csv('test_sequence.csv')
   
    print(f'# 样本数: {len(predict_data)}')
    pep_inputs_a, pep_inputs_b = process_data(predict_data)
    data_loader = Data.DataLoader(PeptideDataset(pep_inputs_a, pep_inputs_b), batch_size, shuffle=False, num_workers=0)
    return pep_inputs_a, pep_inputs_b, data_loader

# 将肽链A和肽链B的序列转为数值向量
def process_data(data):
    pep_inputs_a, pep_inputs_b = [], []
    peptides_a = data['peptide_a']
    peptides_b = data['peptide_b']
    
    for pep_a, pep_b in zip(peptides_a, peptides_b):
        pep_a, pep_b = pep_a.ljust(pep_a_max_len, '-'), pep_b.ljust(pep_b_max_len, '-')
        pep_input_a = [[vocab[n] for n in pep_a]]
        pep_input_b = [[vocab[n] for n in pep_b]]
        pep_inputs_a.extend(pep_input_a)
        pep_inputs_b.extend(pep_input_b)
        
    return torch.LongTensor(pep_inputs_a), torch.LongTensor(pep_inputs_b)

# 定义自定义数据集类，用于封装肽链A和B的数据
class PeptideDataset(Data.Dataset):
    def __init__(self, pep_inputs_a, pep_inputs_b):
        super(PeptideDataset, self).__init__()
        self.pep_inputs_a = pep_inputs_a
        self.pep_inputs_b = pep_inputs_b

    def __len__(self):
        return self.pep_inputs_a.shape[0]

    def __getitem__(self, idx):
        return self.pep_inputs_a[idx], self.pep_inputs_b[idx]


# 位置编码，用于向Transformer模型提供序列的位置信息
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# 注意力机制：缩放点积注意力
class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn


# 多头注意力机制，处理肽链A和肽链B的特征
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        batch_size = input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return output,attn

# 前馈神经网络，用于处理注意力机制的输出
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.fc(x)

# 编码器层，用于处理肽链A和肽链B的特征
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        attn_output, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

# 编码器，负责处理输入序列（肽链A和肽链B）
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        for layer in self.layers:
            enc_outputs, _ = layer(enc_outputs, enc_self_attn_mask)
        return enc_outputs

# 生成掩码函数，用于处理填充
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)





# 解码器
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs):
        """解码器前向传播"""
        dec_outputs = self.pos_emb(dec_inputs.transpose(0, 1)).transpose(0, 1).to(device)
        dec_self_attn_pad_mask = torch.LongTensor(np.zeros((dec_inputs.shape[0], tgt_len, tgt_len))).bool().to(device)
        dec_self_attns = []
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_pad_mask)
            dec_self_attns.append(dec_self_attn)
        return dec_outputs, dec_self_attns
    


# Transformer模型，用于预测肽链A和肽链B的结合位点
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=1, d_ff=256, n_heads=8):
        super(Transformer, self).__init__()
        self.encoder_a = Encoder(vocab_size, d_model, n_layers, d_ff, n_heads)
        self.encoder_b = Encoder(vocab_size, d_model, n_layers, d_ff, n_heads)
        self.fc = nn.Linear(d_model, 1)  # 输出预测结果

    def forward(self, input_a, input_b):
        enc_output_a = self.encoder_a(input_a)
        enc_output_b = self.encoder_b(input_b)
        combined_output = enc_output_a * enc_output_b  # 将两个编码器的输出结合
        output = self.fc(combined_output.mean(dim=1))  # 进行平均池化并输出
        return torch.sigmoid(output)  # 返回0到1之间的预测概率

# 模型评估函数
def eval_step(model, val_loader, threshold=0.5, use_cuda=False):
    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()
    with torch.no_grad():
        y_prob_val_list, dec_attns_val_list = [], []
        for val_pep_inputs, val_pep_b_inputs in val_loader:
            val_pep_inputs, val_pep_b_inputs = val_pep_inputs.to(device), val_pep_b_inputs.to(device)
            val_outputs, _, _, val_dec_self_attns = model(val_pep_inputs, val_pep_b_inputs)
            y_prob_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach().numpy()
            y_prob_val_list.extend(y_prob_val)
            dec_attns_val_list.extend(val_dec_self_attns[0][:, :, pep_max_len:, :pep_max_len])
        y_pred_val_list = transfer(y_prob_val_list, threshold)
    return y_pred_val_list, y_prob_val_list, dec_attns_val_list


# 测试函数
def test_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for seq_a, seq_b, labels in dataloader:
            outputs = model(seq_a, seq_b)
            predicted = (outputs.squeeze() > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f"Accuracy: {100 * correct / total:.2f}%")

