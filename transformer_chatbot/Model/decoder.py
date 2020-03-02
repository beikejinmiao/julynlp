import copy
import torch
from torch import nn
from transformer_chatbot.Net.decoderLayer import DecoderLayer
from transformer_chatbot.Net.embedding import Embedder
from transformer_chatbot.Net.norm import Norm
from transformer_chatbot.Unit.positionalEncoder import PositionalEncoder


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, device, weight_matrix):
        super().__init__()
        self.N = N
        self.embed = Embedder(weight_matrix).to(device)
        self.linear = nn.Linear(weight_matrix.shape[1], d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = self.get_clones(DecoderLayer(d_model, heads, 0.3), N)
        self.norm = Norm(d_model)

    def get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)  ## [batchsize, max_seq_len,embed_len]
        x = torch.tanh(self.linear(x)) # [batchsize, max_seq_len, d_model]
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)  # [batchsize, max_seq_len, d_model]
