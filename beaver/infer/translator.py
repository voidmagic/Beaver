# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class Translator(nn.Module):
    def __init__(self, opt, model, fields):
        self.beam_size = opt.beam_size
        self.eos = fields["tgt"].eos_id
        self.bos = fields["tgt"].bos_id
        self.pad = fields["tgt"].pad_id
        self.num_words = model.generator.vocab_size
        self.enc_pad = model.encoder.embedding.word_padding_idx
        self.max_length = opt.max_length

        super(Translator, self).__init__()

        self.register_buffer("inf", torch.tensor(-1e20).float())
        self.register_buffer("length_penalty", (6 / (5 + torch.arange(self.max_length).float())) ** opt.length_penalty)
        self.register_buffer("pad_tensor", torch.full([1], self.pad).long())
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.generator = model.generator

    def forward(self, src):
        batch_size = src.size(0)
        device = src.device

        src_pad = src.eq(self.enc_pad)
        enc_out = self.repeat_beam(self.encoder(src, src_pad))
        src_pad = self.repeat_beam(src_pad)

        previous = None

        alive_hypotheses = torch.full([batch_size * self.beam_size, 1], fill_value=self.pad).long()
        alive_hypotheses[::4, :] = self.bos
        alive_scores = torch.full([batch_size, self.beam_size], -1e20).float()
        alive_scores[:, 0] = 0.

        beam_index_expander = torch.arange(batch_size).unsqueeze(1) * self.beam_size

        alive_hypotheses = alive_hypotheses.to(device)
        alive_scores = alive_scores.to(device)
        beam_index_expander = beam_index_expander.to(device)

        finished = [[] for _ in range(batch_size)]

        for i in range(self.max_length-2):
            # [batch_size x beam_size, 1]
            current_token = alive_hypotheses[:, -1:]
            tgt_pad = current_token.eq(self.pad)
            dec_out, previous = self.decoder(current_token, enc_out, src_pad, tgt_pad, previous, i)

            # [batch x beam, vocab]
            out = self.generator(dec_out).view(batch_size, self.beam_size, -1)
            beam_scores = alive_scores.unsqueeze(-1) + out

            # 1. select 2 x beam candidates
            # [batch, 2 x beam]
            top_scores, index = beam_scores.view(batch_size, -1).topk(self.beam_size)
            origin = index // self.num_words
            tokens = index % self.num_words

            origin = (origin + beam_index_expander).view(-1)
            candidate_seqs = alive_hypotheses.index_select(0, origin)
            candidate_seqs = torch.cat([candidate_seqs, tokens.view(-1, 1)], dim=-1)
            previous = previous.index_select(0, origin.view(-1))

            # 2. pick out unfinished sequences
            # [batch, 2 x beam]
            flags = torch.eq(tokens, self.eos)
            alive_masked_scores = top_scores + flags.float() * self.inf
            finish_masked_scores = top_scores + (1 - flags.float()) * self.inf

            # 3. select beam unfinished candidates
            alive_scores, alive_indices = alive_masked_scores.topk(self.beam_size)
            alive_indices = (alive_indices + beam_index_expander).view(-1)
            alive_hypotheses = candidate_seqs.index_select(0, alive_indices.view(-1))
            previous = previous.index_select(0, alive_indices.view(-1))

            # 4. select finished sequences
            for j in range(batch_size):
                for idx, tok in enumerate(candidate_seqs.view(batch_size, self.beam_size, -1)[j, :, -1]):
                    if tok == self.eos:
                        finished[j].append(
                            (top_scores[j, idx].clone(), candidate_seqs.view(batch_size, self.beam_size, -1)[j, idx, 1:]))

            # end condition
            max_score, _ = torch.max(alive_scores.view(batch_size, -1) * self.length_penalty[i + 2], dim=-1)
            max_finish = [max([t[0] * self.length_penalty[t[1].size(0)] for t in finished[j]]) if finished[j] else self.inf for j
                          in range(batch_size)]
            max_finish = torch.stack(max_finish)
            if torch.sum(max_finish < max_score) == 0:
                break

        result = []
        for j in range(batch_size):
            if not finished[j]:
                result.append(alive_hypotheses[j, 0, 1:])
            else:
                candidate = sorted(finished[j], key=lambda t: t[0] * self.length_penalty[t[1].size(0)], reverse=True)
                result.append(candidate[0][1])
        return torch.stack([t for t in self.pack_result(result)], dim=0)

    def repeat_beam(self, tensor):
        dim = tensor.dim()
        size = tensor.size()
        tensor = tensor.unsqueeze(1).repeat(1, self.beam_size, *[1 for _ in range(dim - 1)]).view(self.beam_size*size[0], *size[1:])
        return tensor

    def pack_result(self, result):
        for r in result:
            pad_len = self.max_length - r.size(0)
            yield r if pad_len == 0 else torch.cat([r, self.pad_tensor.repeat(pad_len)], dim=0)
