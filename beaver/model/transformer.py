# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, hidden_size, inner_size, dropout=0.0):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, inner_size)
        self.w_2 = nn.Linear(inner_size, hidden_size)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)
        nn.init.constant_(self.w_1.bias, 0.)
        nn.init.constant_(self.w_2.bias, 0.)

    def forward(self, x):
        y = self.w_1(x)
        y = self.relu(y)
        y = self.dropout_1(y)
        y = self.w_2(y)
        return y


class EncoderLayer(nn.Module):

    def __init__(self, hidden_size, dropout, head_count, ff_size):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(head_count, hidden_size, dropout=dropout)
        self.feed_forward = FeedForward(hidden_size, ff_size, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(2)])

    def forward(self, x, mask):
        # self attention
        y = self.self_attn(x, mask=mask)
        x = self.norm[0](x + self.dropout(y))

        # feed forward
        y = self.feed_forward(x)
        x = self.norm[1](x + self.dropout(y))
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, num_heads, hidden_size, dropout, ff_size, embedding):
        super(Encoder, self).__init__()
        self.embedding = embedding
        self.layers = nn.ModuleList([EncoderLayer(hidden_size, dropout, num_heads, ff_size) for _ in range(num_layers)])

    def forward(self, src, src_pad):
        src_mask = src_pad.unsqueeze(1).repeat(1, src.size(1), 1)
        output = self.embedding(src)
        for layer in self.layers:
            output = layer(output, src_mask)
        return output


class DecoderLayer(nn.Module):

    def __init__(self, hidden_size, dropout, head_count, ff_size):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(head_count, hidden_size, dropout=dropout)
        self.src_attn = MultiHeadedAttention(head_count, hidden_size, dropout=dropout)
        self.feed_forward = FeedForward(hidden_size, ff_size, dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask, previous=None):
        all_input = x if previous is None else torch.cat((previous, x), dim=1)

        # self attention
        y = self.self_attn(x, all_input, tgt_mask)
        x = self.norm[0](x + self.dropout(y))

        # encoder decoder attention
        y = self.src_attn(x, enc_out, src_mask)
        x = self.norm[1](x + self.dropout(y))

        # feed forward
        y = self.feed_forward(x)
        x = self.norm[2](x + self.dropout(y))
        return x, all_input


class Decoder(nn.Module):

    def __init__(self, num_layers, num_heads, hidden_size, dropout, ff_size, embedding):
        super(Decoder, self).__init__()
        self.embedding = embedding
        self.layers = nn.ModuleList([DecoderLayer(hidden_size, dropout, num_heads, ff_size) for _ in range(num_layers)])
        self.register_buffer("upper_triangle", torch.triu(torch.ones(1000, 1000), diagonal=1).byte())

    def forward(self, tgt, enc_out, src_pad, tgt_pad, previous=None, timestep=0):

        output = self.embedding(tgt, timestep)
        tgt_len = tgt.size(1)

        src_mask = src_pad.unsqueeze(1).repeat(1, tgt_len, 1)
        tgt_mask = tgt_pad.unsqueeze(1).repeat(1, tgt_len, 1)
        upper_triangle = self.upper_triangle[:tgt_len, :tgt_len]
        # tgt mask: 0 if not upper and not pad, 1 or 2 otherwise

        tgt_mask = torch.gt(tgt_mask + upper_triangle, 0)
        saved_inputs = []
        for layer in self.layers:
            prev_layer = None if previous is None else previous[:, i]
            tgt_mask = tgt_mask if previous is None else None

            output, all_input = layer(output, enc_out, src_mask, tgt_mask, prev_layer)
            saved_inputs.append(all_input)
        return output, torch.stack(saved_inputs, dim=1)


class MultiHeadedAttention(nn.Module):

    def __init__(self, head_count, model_dim, dropout=0.0):
        self.dim_per_head = model_dim // head_count
        self.head_count = head_count

        super(MultiHeadedAttention, self).__init__()

        self.linear_keys = nn.Linear(model_dim, model_dim)
        self.linear_values = nn.Linear(model_dim, model_dim)
        self.linear_query = nn.Linear(model_dim, model_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_keys.weight)
        nn.init.xavier_uniform_(self.linear_query.weight)
        nn.init.xavier_uniform_(self.linear_values.weight)
        nn.init.xavier_uniform_(self.final_linear.weight)
        nn.init.constant_(self.linear_keys.bias, 0.)
        nn.init.constant_(self.linear_query.bias, 0.)
        nn.init.constant_(self.linear_values.bias, 0.)
        nn.init.constant_(self.final_linear.bias, 0.)

    def forward(self, query, memory=None, mask=None):
        memory = query if memory is None else memory
        batch_size = memory.size(0)

        def split_head(x):
            return x.view(batch_size, -1, self.head_count, self.dim_per_head).transpose(1, 2)

        def combine_head(x):
            return x.transpose(1, 2).contiguous().view(batch_size, -1, self.head_count * self.dim_per_head)

        # 1) Project key, value, and query.
        key = split_head(self.linear_keys(memory))
        value = split_head(self.linear_values(memory))
        query = split_head(self.linear_query(query))

        # 2) Calculate and scale scores.
        query = query / math.sqrt(self.dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e20)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = combine_head(torch.matmul(drop_attn, value))

        output = self.final_linear(context)
        return output
