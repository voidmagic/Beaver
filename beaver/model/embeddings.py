# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn


def positional_encoding(embedding_dim, max_len=1e4):
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, embedding_dim // 2).float() * -(math.log(1.e4) / ((embedding_dim // 2) - 1)))
    pe = torch.cat([torch.sin(position * div_term), torch.cos(position * div_term)], dim=1)
    return pe


class Embedding(nn.Module):
    def __init__(self, embedding_dim, vocab_size, padding_idx, dropout):
        self.word_padding_idx = padding_idx
        self.embedding_dim = embedding_dim
        pe = positional_encoding(embedding_dim)
        super(Embedding, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.embedding_dim ** -0.5)
        nn.init.constant_(self.embedding.weight[self.padding_idx], 0)

    @property
    def padding_idx(self):
        return self.word_padding_idx

    def forward(self, x, timestep=0):
        embedding = self.embedding(x) * (self.embedding_dim ** 0.5) + self.pe[timestep:timestep + x.size(1)]
        return self.dropout(embedding)
