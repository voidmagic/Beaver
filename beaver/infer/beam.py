# -*- coding: utf-8 -*-
import torch


class Beam(object):

    def __init__(self, beam_size, pad, bos, eos, device, lp):

        self.alpha = lp
        self.hypotheses = torch.full([beam_size, 1], fill_value=pad).long().to(device)
        self.finished = []

    @property
    def best_hypothesis(self):
        finished = sorted(self.finished, key=lambda t: self.length_penalty(t[0], t[1].size(0)), reverse=True)
        if not finished:
            return self.hypotheses[0, 1:]

        return finished[0][1]

    def length_penalty(self, score, length):
        return score * ((6 / (5 + length)) ** self.alpha)

    @property
    def done(self):
        max_score = max([self.length_penalty(score, self.hypotheses.size(0)) for score in self.scores])
        max_finish = max([self.length_penalty(t[0], t[1].size(0)) for t in self.finished]) if self.finished else -1e20
        return bool(max_score < max_finish)
