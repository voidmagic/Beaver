# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from beaver.infer.beam import Beam


def beam_search(opt, model, batch, fields, device):
    batch_size = batch.batch_size
    beam_size = opt.beam_size
    num_words = model.generator.vocab_size
    encoder = nn.DataParallel(model.encoder).to(device)
    decoder = nn.DataParallel(model.decoder).to(device)
    generator = nn.DataParallel(model.generator).to(device)

    pad = fields["tgt"].pad_id
    eos = fields["tgt"].eos_id

    beams = [Beam(opt.beam_size, fields["tgt"].pad_id, fields["tgt"].bos_id, fields["tgt"].eos_id,
                  device, opt.length_penalty) for _ in range(batch_size)]

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

    beam_index_expander = torch.arange(batch_size).unsqueeze(1) * beam_size
    length_penalty = (6 / (5 + torch.arange(opt.max_length))) ** opt.length_penalty
    inf = torch.tensor(-1e20).to(device)

    alive_hypotheses = alive_hypotheses.to(device)
    alive_scores = alive_scores.to(device)
    beam_index_expander = beam_index_expander.to(device)
    length_penalty = length_penalty.float().to(device)

    finished = [[] for _ in range(batch_size)]

    for i in range(opt.max_length):
        # [batch_size x beam_size, 1]
        current_token = alive_hypotheses[:, -1:]
        tgt_pad = current_token.eq(model.decoder.embedding.word_padding_idx)
        dec_out, previous = decoder(current_token, enc_out, src_pad, tgt_pad, previous, i)

        # [batch x beam, vocab]
        out = generator(dec_out).view(batch_size, beam_size, -1)
        beam_scores = alive_scores.unsqueeze(-1) + out

        # [batch, beam]
        alive_scores, index = beam_scores.view(batch_size, -1).topk(beam_size)
        origin = index / num_words
        tokens = index - origin * num_words

        origin = (origin + beam_index_expander).view(-1)

        hypotheses = torch.index_select(alive_hypotheses, 0, origin)
        alive_hypotheses = torch.cat([hypotheses, tokens.view(-1).unsqueeze(-1)], dim=-1)
        previous = previous.index_select(0, origin)

        for j, b in enumerate(beams):
            b.scores = alive_scores[j]
            b.hypotheses = alive_hypotheses.view(batch_size, beam_size, -1)[j]
            for idx, tok in enumerate(b.hypotheses[:, -1]):
                if tok == eos:
                    finished[j].append((b.scores[idx].clone(), b.hypotheses[idx, 1:]))
                    b.finished.append((b.scores[idx].clone(), b.hypotheses[idx, 1:]))

        if all((b.done for b in beams)):
            break

    return [b.best_hypothesis for b in beams]

