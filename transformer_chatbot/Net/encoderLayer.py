from torch import nn

from transformer_chatbot.Net.feedForward import FeedForward
from transformer_chatbot.Net.multiHeadAttention import MultiHeadAttention
from transformer_chatbot.Net.norm import Norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.2):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x [batchsize, max_seq_len, d_model]
        #trg_mask 上三角形 [1,seq_len-1,seq_len-1]

        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask)) # [batchsize, max_seq_len, d_model]
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))  # [batchsize, max_seq_len, d_model]
        return x  # [batchsize, max_seq_len, d_model]
