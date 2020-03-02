import math
import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.3):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        # q, k, v = [batchsize, 8, max_seq_len, d_model // heads]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # 权重
        # scores = [batchsize, 8, max_seq_len, max_seq_len]

        if mask is not None:
            mask = mask.unsqueeze(1) #[batch_size,1,1,seq_len]      trg_mask 上三角形 [1,1,seq_len-1,seq_len-1]
            scores = scores.masked_fill(mask == 0, -1e9) # 0的换无限小
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)  # [batchsize, 8, max_seq_len, d_model // heads]
        return output

    def forward(self, q, k, v, mask=None):
        # x = q, k, v [batchsize, max_seq_len, d_model]
        bs = q.size(0) # batchsize

        # // 整数除法

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k) #[batchsize,max_seq_len, d_model] --> [batchsize, max_seq_len, 8, d_model // heads]
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)  #[batchsize, 8, max_seq_len, d_model // heads]
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout) # [batchsize, 8, max_seq_len, d_model // heads]

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model) # [batchsize, max_seq_len, d_model]

        output = self.out(concat)

        return output # [batchsize, max_seq_len, d_model]
