from torch import nn

from transformer_chatbot.Model.decoder import Decoder
from transformer_chatbot.Model.encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, device, en_weight_matrix, de_weight_matrix):
        super().__init__()

        self.encoder = Encoder(src_vocab, d_model, N, heads, device, en_weight_matrix).to(device) #编码器
        self.decoder = Decoder(trg_vocab, d_model, N, heads, device, de_weight_matrix).to(device) #解码器
        self.out = nn.Linear(d_model, trg_vocab).to(device)

    def forward(self, src, trg, src_mask, trg_mask):
        # src_vocab [batch_size,seq_len]   src_mask #[batch_size,1,seq_len]
        e_outputs = self.encoder(src, src_mask) # [batchsize,max_seq_len,d_model]
        ##trg 去除了最后一个 [batch_size,seq_len-1]   src_mask #[batch_size,1,seq_len] trg_mask 上三角形 [1,seq_len-1,seq_len-1]
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)  #[batchsize,max_seq_len,trg_vocab]
        return output