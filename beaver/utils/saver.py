
import torch
import os
import datetime


class Saver(object):
    def __init__(self, save_path, max_to_keep):
        self.ckpt_names = []
        self.save_path = save_path + datetime.datetime.now().strftime("-%y%m%d-%H%M%S")
        self.max_to_keep = max_to_keep
        os.mkdir(self.save_path)

    def save(self, save_dict, opt_str, step, bleu, loss):
        filename = "checkpoint-step-%06d" % step
        full_filename = os.path.join(self.save_path, filename)
        self.ckpt_names.append(full_filename)
        torch.save(save_dict, full_filename)

        with open(os.path.join(self.save_path, "log"), "a", encoding="UTF-8") as log:
            log.write("%s\t step: %6d\t loss: %.2f\t bleu: %.2f\n" % (datetime.datetime.now(), step, loss, bleu))
        with open(os.path.join(self.save_path, "params"), "a", encoding="UTF-8") as log:
            log.write(opt_str + "\n")

        if 0 < self.max_to_keep < len(self.ckpt_names):
            earliest_ckpt = self.ckpt_names.pop(0)
            os.remove(earliest_ckpt)


class Loader(object):
    def __init__(self, save_path, params, logger):
        self.path = save_path
        self.empty = self.check_empty()
        self.logger = logger

        self.checkpoint = self.load_checkpoint() if not self.empty else None
        self.step = self.get_step() if not self.empty else 0
        self.params = self.checkpoint["opt"] if not self.empty else params

    def check_empty(self):
        if os.path.exists(self.path) and os.path.isdir(self.path):
            fs = [f for f in os.listdir(self.path) if f.startswith("checkpoint")]
            if len(fs) > 0:
                return False
        return True

    def get_step(self):
        fs = [f for f in os.listdir(self.path) if f.startswith("checkpoint")]
        return int(sorted(fs, reverse=True)[0].split("-")[-1])

    def load_checkpoint(self):
        fs = [f for f in os.listdir(self.path) if f.startswith("checkpoint")]
        f = os.path.join(self.path, sorted(fs, reverse=True)[0])
        self.logger.info("Load checkpoint from %s." % f)
        return torch.load(f, map_location=lambda storage, loc: storage)
