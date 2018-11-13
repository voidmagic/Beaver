# -*- coding: utf-8 -*-

import logging

import torch.cuda

from .metric import calculate_bleu, file_bleu
from .saver import Saver, Loader


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
