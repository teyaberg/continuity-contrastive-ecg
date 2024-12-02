from continuity_contrastive_ecg.models.base_model import BaseModel


class ConstrastiveModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _step(self, batch, batch_idx=-1, mode="train"):
        x1, x2 = batch
        z1 = self.forward(x1)
        z2 = self.forward(x2)
        loss = self.loss(z1, z2)
        return loss
