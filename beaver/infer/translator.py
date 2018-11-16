# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


def beam_search(opt, model, batch, fields, device):
    batch_size = batch.batch_size
    beam_size = opt.beam_size
    num_words = model.generator.vocab_size
    encoder = nn.DataParallel(model.encoder).to(device)
    decoder = nn.DataParallel(model.decoder).to(device)
    generator = nn.DataParallel(model.generator).to(device)

    pad = fields["tgt"].pad_id
    eos = fields["tgt"].eos_id

    def repeat_beam(tensor):
        dim = tensor.dim()
        size = tensor.size()
        tensor = tensor.unsqueeze(1).repeat(1, beam_size, *[1 for _ in range(dim - 1)]).view(beam_size*size[0], *size[1:])
        return tensor

    src_pad = batch.src.eq(model.encoder.embedding.word_padding_idx)
    enc_out = repeat_beam(encoder(batch.src, src_pad))
    src_pad = repeat_beam(src_pad)

    previous = None

    alive_hypotheses = torch.full([batch_size * beam_size, 1], fill_value=pad).long()
    alive_hypotheses[::4, :] = fields["tgt"].bos_id
    alive_scores = torch.full([batch_size, beam_size], -1e20).float()
    alive_scores[:, 0] = 0.

    finish_hypotheses = torch.full([batch_size * beam_size, 1], fill_value=pad).long()
    finish_scores = torch.full([batch_size, beam_size], -1e20).float()

    beam_index_expander = torch.arange(batch_size).unsqueeze(1) * beam_size
    length_penalty = (6 / (5 + torch.arange(opt.max_length+2).float())) ** opt.length_penalty
    inf = torch.tensor(-1e20).float().to(device)

    alive_hypotheses = alive_hypotheses.to(device)
    alive_scores = alive_scores.to(device)
    beam_index_expander = beam_index_expander.to(device)
    length_penalty = length_penalty.to(device)

    finish_hypotheses = finish_hypotheses.to(device)
    finish_scores = finish_scores.to(device)

    finished = [[] for _ in range(batch_size)]

    for i in range(opt.max_length):
        # [batch_size x beam_size, 1]
        current_token = alive_hypotheses[:, -1:]
        tgt_pad = current_token.eq(model.decoder.embedding.word_padding_idx)
        dec_out, previous = decoder(current_token, enc_out, src_pad, tgt_pad, previous, i)

        # [batch x beam, vocab]
        out = generator(dec_out).view(batch_size, beam_size, -1)
        beam_scores = alive_scores.unsqueeze(-1) + out

        # 1. select 2 x beam candidates
        # [batch, 2 x beam]
        top_scores, index = beam_scores.view(batch_size, -1).topk(beam_size)
        origin = index // num_words
        tokens = index % num_words

        origin = (origin + beam_index_expander).view(-1)
        candidate_seqs = alive_hypotheses.index_select(0, origin)
        candidate_seqs = torch.cat([candidate_seqs, tokens.view(-1, 1)], dim=-1)
        previous = previous.index_select(0, origin.view(-1))

        # 2. pick out unfinished sequences
        # [batch, 2 x beam]
        flags = torch.eq(tokens, eos)
        alive_masked_scores = top_scores + flags.float() * inf
        finish_masked_scores = top_scores + (1 - flags.float()) * inf

        # 3. select beam unfinished candidates
        alive_scores, alive_indices = alive_masked_scores.topk(beam_size)
        alive_indices = (alive_indices + beam_index_expander).view(-1)
        alive_hypotheses = candidate_seqs.index_select(0, alive_indices.view(-1))
        previous = previous.index_select(0, alive_indices.view(-1))

        # 4. select finished sequences
        for j in range(batch_size):
            for idx, tok in enumerate(candidate_seqs.view(batch_size, beam_size, -1)[j, :, -1]):
                if tok == eos:
                    finished[j].append((top_scores[j, idx].clone(), candidate_seqs.view(batch_size, beam_size, -1)[j, idx, 1:]))

        # end condition
        max_score, _ = torch.max(alive_scores.view(batch_size, -1) * length_penalty[i+2], dim=-1)
        max_finish = [max([t[0] * length_penalty[t[1].size(0)] for t in finished[j]]) if finished[j] else inf for j in range(batch_size)]
        max_finish = torch.stack(max_finish)
        if torch.sum(max_finish < max_score) == 0:
            break

    result = []
    for j in range(batch_size):
        if not finished[j]:
            result.append(alive_hypotheses[j, 0, 1:])
        else:
            candidate = sorted(finished[j], key=lambda t: t[0] * length_penalty[t[1].size(0)], reverse=True)
            result.append(candidate[0][1])
    return result

