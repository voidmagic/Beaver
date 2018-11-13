
import torch
import os
import datetime


class Saver(object):
    def __init__(self, save_path, max_to_keep, logger):
        self.ckpt_names = []
        self.save_path = save_path
        self.max_to_keep = max_to_keep
        self.bleu_logs = []
        if os.path.exists(save_path) and not os.path.isdir(save_path):
            logger.info("%s is not a valid path" % save_path)
            exit()
        elif not os.path.exists(save_path):
            os.mkdir(save_path)

    def save(self, save_dict, step, bleu):
        filename = "checkpoint-step-%06d" % step
        full_filename = os.path.join(self.save_path, filename)
        self.ckpt_names.append(full_filename)
        torch.save(save_dict, full_filename)

        self.bleu_logs.append("%s\t step: %6d\t bleu: %.2f\n" % (datetime.datetime.now(), step, bleu))
        with open(os.path.join(self.save_path, "log"), "w", encoding="UTF-8") as log:
            log.writelines(self.bleu_logs)

        if 0 < self.max_to_keep < len(self.ckpt_names):
            earliest_ckpt = self.ckpt_names.pop(0)
            os.remove(earliest_ckpt)


class Loader(object):
    def __init__(self, save_path, params):
        self.path = save_path
        self.params = params

    @property
    def last_model(self):
        if os.path.exists(self.path) and os.path.isdir(self.path):
            fs = [f for f in os.listdir(self.path) if f.startswith("checkpoint")]
            if len(fs) > 0:
                return os.path.join(self.path, sorted(fs, reverse=True)[0])
        return None
