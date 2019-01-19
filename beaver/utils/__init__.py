# -*- coding: utf-8 -*-

import logging

import torch.cuda

from beaver.utils.metric import calculate_bleu, file_bleu
from beaver.utils.saver import Saver, Loader


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_logger():
    log_format = logging.Formatter("%(asctime)s - %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    return logger


def printing_opt(opt):
    return "\n".join(["%15s | %s" % (e[0], e[1]) for e in sorted(vars(opt).items(), key=lambda x: x[0])])
