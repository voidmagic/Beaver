# -*- coding: utf-8 -*-

import argparse

import torch
import torch.cuda
import torch.nn as nn

from beaver.data import build_dataset
from beaver.infer import parallel_beam_search
from beaver.loss import WarmAdam, LabelSmoothingLoss
from beaver.model import NMTModel, FullModel
from beaver.utils import Saver, Loader
from beaver.utils import calculate_bleu
from beaver.utils import parseopt, get_device, get_logger

parser = argparse.ArgumentParser()

parseopt.data_opts(parser)
parseopt.train_opts(parser)
parseopt.model_opts(parser)

opt = parser.parse_args()

device = get_device()
logger = get_logger()

saver = Saver(save_path=opt.model_path, max_to_keep=opt.max_to_keep, logger=logger)
loader = Loader(opt.model_path, opt, logger)


def valid(model, valid_dataset):
    model.eval()
    total_loss, total = 0.0, 0

    hypothesis, references = [], []

    for batch in valid_dataset:
        loss = model(batch.src, batch.tgt).mean()
        total_loss += loss.data
        total += 1

        predictions = parallel_beam_search(opt, model.module.model, batch, valid_dataset.fields)
        hypothesis += [valid_dataset.fields["tgt"].decode(p) for p in predictions]
        references += [valid_dataset.fields["tgt"].decode(t) for t in batch.tgt]

    bleu = calculate_bleu(hypothesis, references)
    logger.info("Valid loss: %.2f\tValid Beam BLEU: %3.2f" % (total_loss / total, bleu))
    return total_loss / total, bleu


def train(model, optimizer, train_dataset, valid_dataset):
    total_loss = 0.0
    model.zero_grad()
    for i, batch in enumerate(train_dataset):
        loss = model(batch.src, batch.tgt).mean()
        loss.backward()
        total_loss += loss.data

        if (i + 1) % opt.grad_accum == 0:
            optimizer.step()
            model.zero_grad()

            if optimizer.n_step % opt.report_every == 0:
                logger.info("step: %7d\t loss: %7f" % (optimizer.n_step, total_loss / opt.report_every / opt.grad_accum))
                total_loss = 0.0

            if optimizer.n_step % opt.save_every == 0:
                with torch.set_grad_enabled(False):
                    valid_loss, valid_bleu = valid(model, valid_dataset)
                    checkpoint = {"model": model.module.model.state_dict(),
                                  "opt": opt}
                    saver.save(checkpoint, optimizer.n_step, valid_bleu, valid_loss)
                model.train()
        del loss


def main():
    logger.info("Build dataset...")
    train_dataset = build_dataset(opt, opt.train, opt.vocab, device, train=True)
    valid_dataset = build_dataset(opt, opt.valid, opt.vocab, device, train=False)
    fields = valid_dataset.fields = train_dataset.fields
    logger.info("Build model...")

    model = NMTModel.load_model(loader, fields)
    criterion = LabelSmoothingLoss(opt.label_smoothing, len(fields["tgt"].vocab), fields["tgt"].pad_id)
    model = nn.DataParallel(FullModel(model, criterion)).to(device)

    optimizer = WarmAdam(model.module.model.parameters(), opt.lr, opt.betas,
                         opt.eps, opt.hidden_size, opt.warm_up, loader.step)

    logger.info("start training...")
    train(model, optimizer, train_dataset, valid_dataset)


if __name__ == '__main__':
    logger.info("\n" + "\n".join(["%15s | %s" % (e[0], e[1]) for e in sorted(vars(opt).items(), key=lambda x: x[0])]))
    main()
