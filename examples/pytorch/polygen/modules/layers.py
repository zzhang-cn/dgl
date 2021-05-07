import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch.distributions import Categorical

class VertexGenerator(nn.Module):
    '''
    Generate next token from the representation. This part is separated from the decoder, mostly for the convenience of sharing weight between embedding and generator.
    log(softmax(Wx + b))
    '''
    def __init__(self, dim_model, vocab_size):
        super(VertexGenerator, self).__init__()
        self.proj = nn.Linear(dim_model, vocab_size)
    
    def forward(self, x):
        return th.log_softmax(
            self.proj(x), dim=-1
        )

class NucleusSamplingGenerator(nn.Module):
    '''
    Nucleus Sampler: keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    '''
    def __init__(self, dim_model, vocab_size=-1, cumulative_p = 0.9, min_tokens_to_keep = 1):
        '''
        args:
            dim_model: input feature dimention.
            vocab_size: vocabulary size.
            cumulative_p: cumulative probability threshold for nucleus sampling.
            min_tokens_to_keep: 1
        '''
        super(NucleusSamplingGenerator, self).__init__()
        self.proj = None if vocab_size < 0 else nn.Linear(dim_model, vocab_size)
        self.cumulative_p = cumulative_p
        self.min_tokens_to_keep = min_tokens_to_keep
    
    def forward(self, x):
        if self.proj:
            logits = th.softmax(
                self.proj(x), dim=-1
            )
        else:
            logits = x
        # Nucleus Sampling
        sorted_logits, sorted_indices = th.sort(logits, descending=True)
        cumulative_probs = th.cumsum(sorted_logits, dim=-1)
        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.cumulative_p
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :self.min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = 0
        # Normalized the selected probs
        logits = logits * (1.0 / (logits.sum(dim=-1)+1e-6)).unsqueeze(-1)
        # Sampling
        sample_results = Categorical(logits).sample()
        return sample_results


class SubLayerWrapper(nn.Module):
    '''
    The module wraps normalization, dropout, residual connection into one equation:
    sublayerwrapper(sublayer)(x) = x + dropout(sublayer(norm(x)))
    '''
    def __init__(self, size, dropout):
        super(SubLayerWrapper, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    '''
    This module implements feed-forward network(after the Multi-Head Network) equation:
    FFN(x) = max(0, x @ W_1 + b_1) @ W_2 + b_2
    '''
    def __init__(self, dim_model, dim_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_model, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(th.relu(self.w_1(x))))


import copy
def clones(module, k):
    return nn.ModuleList(
        copy.deepcopy(module) for _ in range(k)
    )

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn # (key, query, value, mask)
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayerWrapper(size, dropout), 2)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn # (key, query, value, mask)
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayerWrapper(size, dropout), 3)
