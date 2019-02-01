# -*- coding: utf-8 -*-

import argparse

import torch

from beaver.data import build_dataset
from beaver.infer import parallel_beam_search
from beaver.model import NMTModel
from beaver.utils import parseopt, get_device, get_logger, calculate_bleu, Loader

parser = argparse.ArgumentParser()

parseopt.translate_opts(parser)
parseopt.model_opts(parser)

opt = parser.parse_args()

device = get_device()
logger = get_logger()

loader = Loader(opt.model_path, opt, logger)


def translate(dataset, fields, model):

    total = len(dataset.examples)
    already, hypothesis, references = 0, [], []

    for batch in dataset:
        predictions = parallel_beam_search(opt, model, batch, fields)
        hypothesis += [fields["tgt"].decode(p) for p in predictions]
        already += len(predictions)
        logger.info("Translated: %7d/%7d" % (already, total))

    origin = sorted(zip(hypothesis, dataset.seed), key=lambda t: t[1])
    hypothesis = [h for h, _ in origin]
    with open(opt.output, "w", encoding="UTF-8") as out_file:
        out_file.write("\n".join(hypothesis))
        out_file.write("\n")

    logger.info("Translation finished. ")


def main():
    logger.info("Build dataset...")
    dataset = build_dataset(opt, [opt.input, opt.input], opt.vocab, device, train=False)

    logger.info("Build model...")
    model = NMTModel.load_model(loader, dataset.fields).to(device).eval()

    logger.info("Start translation...")
    with torch.set_grad_enabled(False):
        translate(dataset, dataset.fields, model)


if __name__ == '__main__':
    main()

