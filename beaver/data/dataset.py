# -*- coding: utf-8 -*-

import random
from collections import namedtuple

Batch = namedtuple("Batch", ['src', 'tgt', 'batch_size'])
Example = namedtuple("Example", ['src', 'tgt'])


class TranslationDataset(object):

    def __init__(self, src_path, tgt_path, batch_size, device, train, fields):
        self.batch_size = batch_size
        self.train = train
        self.device = device
        self.fields = fields
        self.sort_key = lambda ex: (len(ex.src), len(ex.tgt))

        examples = []
        for src_line, tgt_line in zip(read_file(src_path), read_file(tgt_path)):
            examples.append(Example(src_line, tgt_line))

        self.examples, self.seed = self.sort(examples)
        self.batches = list(batch(self.examples, self.batch_size))

    def __iter__(self):
        while True:
            for minibatch in self.batches:
                src = self.fields["src"].process([x.src for x in minibatch], self.device)
                tgt = self.fields["tgt"].process([x.tgt for x in minibatch], self.device)
                yield Batch(src=src, tgt=tgt, batch_size=len(minibatch))

            if self.train:
                random.shuffle(self.batches)
            else:
                break

    def sort(self, examples):
        seed = sorted(range(len(examples)), key=lambda idx: self.sort_key(examples[idx]))
        return sorted(examples, key=self.sort_key), seed


def read_file(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.strip().split()


def batch(data, batch_size):
    max_len = 0

    def batch_size_fn(example, count, size):
        nonlocal max_len
        max_len = max(max_len, len(example.src), len(example.tgt))
        return max_len * count

    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
        if size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far, max_len = [ex], batch_size_fn(ex, 1, 0), 0
    if minibatch:
        yield minibatch

