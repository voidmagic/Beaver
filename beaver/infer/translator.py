# -*- coding: utf-8 -*-
import threading

import torch
from torch.nn.parallel import replicate
from torch.nn.parallel.scatter_gather import scatter

from beaver.infer.beam import Beam


def beam_search(opt, model, src, fields, device_idx=0, results=None):
    batch_size = src.size(0)
    beam_size = opt.beam_size
    device = src.device
    num_words = model.generator.vocab_size

    encoder = model.encoder
    decoder = model.decoder
    generator = model.generator

    beams = [Beam(opt.beam_size, fields["tgt"].pad_id, fields["tgt"].bos_id, fields["tgt"].eos_id,
                  device, opt.length_penalty) for _ in range(batch_size)]

    src = src.repeat(1, beam_size).view(batch_size*beam_size, -1)
    src_pad = src.eq(fields["src"].pad_id)
    src_out = encoder(src, src_pad)

    src_length = src.size(1)

    beam_expander = (torch.arange(batch_size) * beam_size).view(-1, 1).to(device)

    previous = None
    max_len = max(int(opt.max_length_ratio * src_length), 10)

    for i in range(max_len):
        if all((b.done for b in beams)):
            break

        # [batch_size x beam_size, 1]
        current_token = torch.cat([b.current_state for b in beams]).unsqueeze(-1)
        tgt_pad = current_token.eq(fields["tgt"].pad_id)
        out, previous = decoder(current_token, src_out, src_pad, tgt_pad, previous, i)
        previous_score = torch.stack([b.scores for b in beams]).unsqueeze(-1)
        out = generator(out).view(batch_size, beam_size, -1)

        # find topk candidates
        scores, indexes = (out + previous_score).view(batch_size, -1).topk(beam_size)

        # find origins and token
        origins = (indexes.view(-1) // num_words).view(batch_size, beam_size)
        tokens = (indexes.view(-1) % num_words).view(batch_size, beam_size)

        for j, b in enumerate(beams):
            b.advance(scores[j], origins[j], tokens[j])

        origins = (origins + beam_expander).view(-1)
        previous = torch.index_select(previous, 0, origins)

    results[device_idx] = [b.best_hypothesis for b in beams]


def parallel_beam_search(opt, model, batch, fields):
    device_ids = list(range(torch.cuda.device_count()))
    results = {}

    if len(device_ids) <= 1:
        beam_search(opt, model, batch.src, fields, results=results)
        return results[0]

    # 1. scatter input
    sources = scatter(batch.src, device_ids)
    targets = scatter(batch.tgt, device_ids)

    # 2. replicate model
    replicas = replicate(model, device_ids[:len(sources)])
    assert len(replicas) == len(sources) == len(targets)

    # 3. parallel apply
    threads = [threading.Thread(target=beam_search, args=(opt, model, src, fields, idx, results))
               for idx, (model, src) in enumerate(zip(replicas, sources))]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return [h for i in range(len(replicas)) for h in results[i]]
