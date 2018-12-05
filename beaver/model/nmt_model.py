# -*- coding: utf-8 -*-

import torch.nn as nn

from beaver.model.embeddings import Embedding
from beaver.model.transformer import Decoder, Encoder


class NMTModel(nn.Module):

    def __init__(self, encoder, decoder, generator):
        super(NMTModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, src, tgt):
        src_pad = src.eq(self.encoder.embedding.word_padding_idx)
        tgt_pad = tgt.eq(self.decoder.embedding.word_padding_idx)

        enc_out = self.encoder(src, src_pad)
        decoder_outputs, _ = self.decoder(tgt[:, :-1], enc_out, src_pad, tgt_pad[:, :-1])
        scores = self.generator(decoder_outputs)
        return scores

    @classmethod
    def build_model(cls, model_opt, fields):
        src_embedding = Embedding.make_embedding(model_opt, fields["src"], model_opt.hidden_size)
        if len(model_opt.vocab) == 2:
            tgt_embedding = Embedding.make_embedding(model_opt, fields["tgt"], model_opt.hidden_size)
        else:
            tgt_embedding = src_embedding

        encoder = Encoder(model_opt.layers,
                          model_opt.heads,
                          model_opt.hidden_size,
                          model_opt.dropout,
                          model_opt.ff_size,
                          src_embedding)

        decoder = Decoder(model_opt.layers,
                          model_opt.heads,
                          model_opt.hidden_size,
                          model_opt.dropout,
                          model_opt.ff_size,
                          tgt_embedding)

        generator = Generator(model_opt.hidden_size, len(fields["tgt"].vocab))
        return cls(encoder, decoder, generator)

    @classmethod
    def load_model(cls, loader, fields):
        model = cls.build_model(loader.params, fields)
        if not loader.empty:
            model.load_state_dict(loader.checkpoint['model'])
        return model


class Generator(nn.Module):
    def __init__(self, hidden_size, tgt_vocab_size):
        self.vocab_size = tgt_vocab_size
        super(Generator, self).__init__()
        self.linear_hidden = nn.Linear(hidden_size, tgt_vocab_size)
        self.lsm = nn.LogSoftmax(dim=-1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_hidden.weight)
        nn.init.constant_(self.linear_hidden.bias, 0.)

    def forward(self, dec_out):
        score = self.linear_hidden(dec_out)
        lsm_score = self.lsm(score)
        return lsm_score.view(-1, self.vocab_size)


class FullModel(nn.Module):
    def __init__(self, model, criterion):
        super(FullModel, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, src, tgt):
        scores = self.model(src, tgt)
        loss = self.criterion(scores, tgt[:, 1:].contiguous().view(-1))
        return loss.unsqueeze(0)
