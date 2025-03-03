import torch

from continuity_contrastive_ecg.models.base_model import BaseModel


class SupervisedModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _step(self, batch, batch_idx=-1, mode="train"):
        # ecgs[0], ecgs[1], outcome, mrn, pid, csn
        x, _, y, _, _, _ = batch
        z = self.forward(x)
        loss = self.loss(z, y)
        return loss

    def get_predictions(self, x):
        return torch.sigmoid(self.forward(x))
