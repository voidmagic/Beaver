# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim


class WarmAdam(object):
    def __init__(self, params, lr, hidden_size, warm_up, init_step):
        self.original_lr = lr
        self.n_step = init_step
        self.hidden_size = hidden_size
        self.warm_up_step = warm_up
        self.optimizer = optim.Adam(params, betas=[0.9, 0.98], eps=1e-9)

    def step(self):
        self.n_step += 1
        warm_up = min(self.n_step ** (-0.5), self.n_step * self.warm_up_step ** (-1.5))
        lr = self.original_lr * (self.hidden_size ** (-0.5) * warm_up)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index):
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing
        self.kl_div = nn.KLDivLoss(reduction='sum')

    def forward(self, output, target):
        numel = target.ne(self.padding_idx).float().sum()
        truth = self.one_hot.repeat(target.size(0), 1)
        truth.scatter_(1, target.unsqueeze(1), self.confidence)
        truth = truth.masked_fill((target == self.padding_idx).unsqueeze(1), 0)
        loss = self.kl_div(output, truth)
        return loss / numel

