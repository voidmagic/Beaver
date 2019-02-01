# -*- coding: utf-8 -*-


def data_opts(parser):
    parser.add_argument("-train", type=str, nargs=2, help="Training data")
    parser.add_argument("-valid", type=str, nargs=2, help="Validation data")
    parser.add_argument("-vocab", type=str, nargs="*", help="Vocab file")


def train_opts(parser):
    parser.add_argument("-model_path", default="train", help="Path to save model")
    parser.add_argument("-grad_accum", type=int, default=1, help="Accumulate gradients")
    parser.add_argument("-batch_size", type=int, default=8192, help="Batch size")
    parser.add_argument("-max_to_keep", type=int, default=5, help="How many checkpoints to keep")
    parser.add_argument("-report_every", type=int, default=1000, help="Report every n steps")
    parser.add_argument("-save_every", type=int, default=2000, help="Valid and save model for every n steps")

    # for validation
    parser.add_argument("-beam_size",  type=int, default=4, help="Beam size")
    parser.add_argument("-max_length_ratio", type=float, default=2.0, help="Maximum prediction length")
    parser.add_argument("-length_penalty",  type=float, default=0.6, help="Length penalty")


def model_opts(parser):
    parser.add_argument("-layers", type=int, default=6, help="Number of layers")
    parser.add_argument("-heads", type=int, default=8, help="Number of heads")
    parser.add_argument("-hidden_size", type=int, default=512, help="Size of hidden states")
    parser.add_argument("-ff_size", type=int, default=2048, help="Feed forward hidden size")

    parser.add_argument("-lr", type=float, default=1.0, help="Learning rate")
    parser.add_argument("-warm_up", type=int, default=8000, help="Warm up step")
    parser.add_argument("-label_smoothing", type=float, default=0.1, help="Label smoothing rate")
    parser.add_argument("-dropout", type=float, default=0.1, help="Dropout rate")


def translate_opts(parser):
    parser.add_argument("-input", type=str, help="Translation data")
    parser.add_argument("-vocab", type=str, nargs="*", help="Vocab file")
    parser.add_argument("-output", default="output.txt", help="Path to output the predictions")

    parser.add_argument("-batch_size", type=int, default=8192, help="Batch size")
    parser.add_argument("-model_path", default="train", help="Path to model checkpoint file")
    parser.add_argument("-beam_size",  type=int, default=4, help="Beam size")
    parser.add_argument("-max_length_ratio", type=float, default=2.0, help="Maximum prediction length")
    parser.add_argument("-length_penalty",  type=float, default=0.6, help="Length penalty")
