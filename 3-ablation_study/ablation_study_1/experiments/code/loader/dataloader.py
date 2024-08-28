import numpy
import random
import itertools
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler


class SideChannel(DataLoader):
    def __init__(self, dataset, config):
        self.init_kwags = {
            'dataset': dataset,
            'drop_last': config.drop_last,
            'batch_size': config.batch_size,
            'shuffle': config.shuffle,
            'num_workers': config.num_workers,
            'pin_memory': True,
        }

        super(SideChannel, self).__init__(**self.init_kwags)

