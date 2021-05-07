import torch as th
import torch.nn as nn
import numpy as np
from .layers import clones

class PositionalEncoding(nn.Module):
    "Position Encoding module"
    def __init__(self, dim_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = th.zeros(max_len, dim_model, dtype=th.float)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, dim_model, 2, dtype=th.float) *
                             -(np.log(10000.0) / dim_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # Not a parameter but should be in state_dict

    def forward(self, pos):
        return th.index_select(self.pe, 1, pos).squeeze(0)


class Embeddings(nn.Module):
    "Word Embedding module"
    def __init__(self, vocab_size, dim_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, dim_model)
        self.dim_model = dim_model

    def forward(self, x):
        return self.lut(x) * np.sqrt(self.dim_model)

class VertCoordJointEmbeddings(nn.Module):
    """ Jointly embed x,y,z coords of a vertext """
    def __init__(self, vocab_size, dim_model):
        super(VertCoordJointEmbeddings, self).__init__()
        self.dim_model = dim_model
        self.num_coords = 3
        self.luts = clones(Embeddings(vocab_size, dim_model), self.num_coords) 
        self.linear = nn.Linear(dim_model*self.num_coords, dim_model) 

    def forward(self, x):
        return self.linear(th.cat((self.luts[0](x[:, 0]), self.luts[1](x[:, 1]), self.luts[2](x[:, 2])), 1))
