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

        self.reps = {
            "train": {"X": [], "y": []},
            "val": {"X": [], "y": []},
            "test": {"X": [], "y": []},
        }
        self.min_val_loss = math.inf

        if hasattr(cfg, "downstream_eval") and cfg.downstream_eval:
            # logger.info("Online downstream task evaluation enabled")
            self.downstream_dataloaders = cfg.downstream_eval.dataloaders
            self.downstream_cfg = cfg.downstream_eval.cfg

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

    def downstream_evaluation(self):
        raise NotImplementedError
