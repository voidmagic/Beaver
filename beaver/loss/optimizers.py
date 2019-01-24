# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.optim as optim


class WarmAdam(object):
    def __init__(self, params, lr, hidden_size, warm_up, init_step):
        self.original_lr = lr
        self.n_step = init_step
        self.hidden_size = hidden_size
        self.warm_up_step = warm_up
        self.params = params
        self.optimizer = optim.Adam(params, betas=[0.9, 0.98], eps=1e-9)

    def step(self):
        self.n_step += 1
        warm_up = min(self.n_step ** (-0.5), self.n_step * self.warm_up_step ** (-1.5))
        lr = self.original_lr * (self.hidden_size ** (-0.5) * warm_up)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        nn.utils.clip_grad_norm_(self.params, 0.1)
        self.optimizer.step()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index):
        self.padding_idx = ignore_index
        self.eps = label_smoothing
        self.v = tgt_vocab_size
        super(LabelSmoothingLoss, self).__init__()

    def forward(self, output, target):
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -output.gather(dim=-1, index=target.view(-1, 1))[non_pad_mask].sum()
        smooth_loss = -output.sum(dim=-1, keepdim=True)[non_pad_mask].sum()
        eps_i = self.eps / self.v
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss / non_pad_mask.float().sum()
