# -*- coding: utf-8 -*-
import torch

from beaver.infer.beam import Beam


def beam_search(opt, model, batch, fields, device):
    batch_size = batch.batch_size
    beam_size = opt.beam_size

    beams = [Beam(opt.beam_size, fields["tgt"].pad_id, fields["tgt"].bos_id, fields["tgt"].eos_id,
                  device, opt.length_penalty) for _ in range(batch_size)]

    src_pad = batch.src.eq(model.encoder.embedding.word_padding_idx)
    enc_out = model.encoder(batch.src, src_pad).repeat(beam_size, 1, 1)
    src_pad = src_pad.repeat(beam_size, 1)

    previous = None

    for i in range(opt.max_length):
        if all((b.done for b in beams)):
            break

        current_token = torch.stack([b.current_state for b in beams], dim=1).view(-1, 1)
        tgt_pad = current_token.eq(model.decoder.embedding.word_padding_idx)
        dec_out, previous = model.decoder(current_token, enc_out, src_pad, tgt_pad, previous, i)

        out = model.generator(dec_out).view(beam_size, batch_size, -1)

        for j, b in enumerate(beams):
            origin = b.advance(out[:, j, :])

            sizes = previous.size()
            p = previous.view(sizes[0], beam_size, -1, sizes[2], sizes[3])[:, :, j]
            p.copy_(p.index_select(1, origin))

    return [b.best_hypothesis for b in beams]

