import math
from abc import ABC, abstractmethod

import lightning as L

from continuity_contrastive_ecg.utils.module import Module


class BaseModel(L.LightningModule, Module, ABC):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = cfg.encoder
        self.projection = cfg.projection
        self.loss = cfg.loss
        self.scheduler = cfg.scheduler
        self.optimizer = cfg.optimizer

        self.min_val_loss = math.inf

    @abstractmethod
    def _step(self, batch, batch_idx=-1, mode="train"):
        pass

    def forward(self, x):
        return self.projection(self.encoder(x))

    def get_representations(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, "val")
        if loss < self.min_val_loss:
            self.min_val_loss = loss
        return loss

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")
