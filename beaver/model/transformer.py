# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size, inner_size, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, inner_size)
        self.w_2 = nn.Linear(inner_size, hidden_size)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

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


class LayerProcess(nn.Module):
    def __init__(self, hidden_dim, mode="none"):
        super(LayerProcess, self).__init__()
        self.mode = mode
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        if self.mode == "layer_norm":
            return x
        else:
            return self.layer_norm(x)


class EncoderLayer(nn.Module):

    def __init__(self, hidden_size, dropout, head_count, ff_size):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(head_count, hidden_size, dropout=dropout)
        self.feed_forward = PositionWiseFeedForward(hidden_size, ff_size, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_preprocess = nn.ModuleList([LayerProcess(hidden_size, "none") for _ in range(2)])
        self.layer_postprocess = nn.ModuleList([LayerProcess(hidden_size, "layer_norm") for _ in range(2)])

    def forward(self, layer_input, mask):
        x = layer_input

        # self attention
        y = self.self_attn(self.layer_preprocess[0](x), mask=mask)
        x = x + self.dropout(y)
        x = self.layer_postprocess[0](x)

        # feed forward
        y = self.feed_forward(self.layer_preprocess[1](x))
        x = x + self.dropout(y)
        x = self.layer_postprocess[1](x)
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, num_heads, hidden_size, dropout, ff_size, embedding):
        self.num_layers = num_layers

        super(Encoder, self).__init__()

        self.embedding = embedding
        self.layers = nn.ModuleList([EncoderLayer(hidden_size, dropout, num_heads, ff_size) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, src, src_pad):
        src_mask = src_pad.unsqueeze(1).repeat(1, src.size(1), 1)

        output = self.embedding(src)
        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask)
        output = self.layer_norm(output)

        return output


class DecoderLayer(nn.Module):

    def __init__(self, hidden_size, dropout, head_count, ff_size):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(head_count, hidden_size, dropout=dropout)
        self.src_attn = MultiHeadedAttention(head_count, hidden_size, dropout=dropout)
        self.feed_forward = PositionWiseFeedForward(hidden_size, ff_size, dropout)
        self.dropout = nn.Dropout(dropout)

        self.layer_preprocess = nn.ModuleList([LayerProcess(hidden_size, "none") for _ in range(3)])
        self.layer_postprocess = nn.ModuleList([LayerProcess(hidden_size, "layer_norm") for _ in range(3)])

    def forward(self, layer_input, enc_out, src_mask, tgt_mask, previous_input=None):
        all_input = layer_input if previous_input is None else torch.cat((previous_input, layer_input), dim=1)
        x = layer_input

        # self attention
        y = self.self_attn(self.layer_preprocess[0](x), self.layer_preprocess[0](all_input), tgt_mask)
        x = x + self.dropout(y)
        x = self.layer_postprocess[0](x)

        # encoder decoder attention
        y = self.src_attn(self.layer_preprocess[1](x), enc_out, src_mask)
        x = x + self.dropout(y)
        x = self.layer_postprocess[1](x)

        # feed forward
        y = self.feed_forward(self.layer_preprocess[2](x))
        x = x + self.dropout(y)
        x = self.layer_postprocess[2](x)
        return x, all_input


class Decoder(nn.Module):

    def __init__(self, num_layers, num_heads, hidden_size, dropout, ff_size, embedding):
        self.num_layers = num_layers
        upper_triangle = torch.triu(torch.ones(1000, 1000), diagonal=1).byte()

        super(Decoder, self).__init__()

        self.embedding = embedding
        self.layers = nn.ModuleList([DecoderLayer(hidden_size, dropout, num_heads, ff_size) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.register_buffer(tensor=upper_triangle, name="upper_triangle")

    def forward(self, tgt, enc_out, src_pad, tgt_pad, previous=None, timestep=0):

        output = self.embedding(tgt, timestep)
        tgt_len = tgt.size(1)

        src_mask = src_pad.unsqueeze(1).repeat(1, tgt_len, 1)
        tgt_mask = tgt_pad.unsqueeze(1).repeat(1, tgt_len, 1)
        upper_triangle = self.upper_triangle[:tgt_len, :tgt_len]
        # tgt mask: 0 if not upper and not pad, 1 or 2 otherwise
        tgt_mask = torch.gt(tgt_mask + upper_triangle, 0)

        saved_inputs = []
        for i in range(self.num_layers):
            prev_layer = None if previous is None else previous[i]
            tgt_mask = tgt_mask if previous is None else None

            output, all_input = self.layers[i](output, enc_out, src_mask, tgt_mask, prev_layer)
            saved_inputs.append(all_input)

        output = self.layer_norm(output)

        return output, torch.stack(saved_inputs)


class MultiHeadedAttention(nn.Module):

    def __init__(self, head_count, model_dim, dropout=0.1):
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.head_count = head_count

        super(MultiHeadedAttention, self).__init__()

        self.linear_keys = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim, head_count * self.dim_per_head)
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
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        key_up = shape(self.linear_keys(memory))
        value_up = shape(self.linear_values(memory))
        query_up = shape(self.linear_query(query))

        # 2) Calculate and scale scores.
        query_up = query_up / math.sqrt(dim_per_head)
        scores = torch.matmul(query_up, key_up.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = unshape(torch.matmul(drop_attn, value_up))

        output = self.final_linear(context)

        return output
