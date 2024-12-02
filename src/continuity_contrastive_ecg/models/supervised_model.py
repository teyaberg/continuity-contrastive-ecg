from continuity_contrastive_ecg.models.base_model import BaseModel


class SupervisedModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _step(self, batch, batch_idx=-1, mode="train"):
        x, y = batch
        z = self.forward(x)
        loss = self.loss(z, y)
        return loss
