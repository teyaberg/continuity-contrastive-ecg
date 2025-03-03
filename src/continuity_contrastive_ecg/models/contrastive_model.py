from continuity_contrastive_ecg.models.base_model import BaseModel


class ContrastiveModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _step(self, batch, batch_idx=-1, mode="train"):
        # ecgs[0], ecgs[1], outcome, mrn, pid, csn
        x1, x2, _, _, _, _ = batch
        # make sure x_1 and x_2 are float tensors
        x1 = x1.float()
        x2 = x2.float()
        z1 = self.forward(x1)
        z2 = self.forward(x2)
        loss = self.loss(z1, z2)
        return loss
