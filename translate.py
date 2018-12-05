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

    total_sents = len(dataset.examples)
    already, hypothesis, references = 0, [], []

    for batch in dataset:
        predictions = parallel_beam_search(opt, model, batch, fields)
        hypothesis += [fields["tgt"].decode(p) for p in predictions]
        references += [fields["tgt"].decode(t) for t in batch.tgt]
        already += len(predictions)
        logger.info("Translated: %7d/%7d" % (already, total_sents))

    origin = sorted(zip(hypothesis, references, dataset.seed), key=lambda t: t[2])
    hypothesis = [o[0] for o in origin]
    references = [o[1] for o in origin]
    with open(opt.output, "w", encoding="UTF-8") as out_file:
        out_file.write("\n".join(hypothesis))

    logger.info("Translation finished. BLEU: %.2f." % calculate_bleu(hypothesis, references))


def main():
    logger.info("Build dataset...")
    dataset = build_dataset(opt, opt.trans, opt.vocab, device, train=False)

    logger.info("Build model...")
    model = NMTModel.load_model(loader, dataset.fields).to(device).eval()

    logger.info("Start translation...")
    with torch.set_grad_enabled(False):
        translate(dataset, dataset.fields, model)


if __name__ == '__main__':
    main()

