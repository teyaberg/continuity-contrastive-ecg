from abc import ABC, abstractmethod

import torch.utils.data as data

from continuity_contrastive_ecg.utils.module import Module


class BaseDataset(data.Dataset, Module, ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    def __scale__(self):
        # implements scaling function
        pass
