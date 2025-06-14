# import torch
# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
from tensorboardX import SummaryWriter

from torch import Tensor
from collections import OrderedDict
import os


class Logger(object):
    def __init__(self, log_dir, tensorboard=True, matplotlib=True):

        self.reset(log_dir, tensorboard, matplotlib)

    def reset(self, log_dir=None, tensorboard=True, matplotlib=True):

        if log_dir is not None: self.log_dir=log_dir 
        self.writer = SummaryWriter(log_dir=self.log_dir) if tensorboard else None
        self.counter = OrderedDict()

    def update_scalers(self, ordered_dict):

        for key, value in ordered_dict.items():
            if isinstance(value, Tensor):
                ordered_dict[key] = value.item()
            if self.counter.get(key) is None:
                self.counter[key] = 1
            else:
                self.counter[key] += 1

            if self.writer:
                self.writer.add_scalar(key, value, self.counter[key])

# implementation follows https://github.com/PatrickHua/SimSiam




