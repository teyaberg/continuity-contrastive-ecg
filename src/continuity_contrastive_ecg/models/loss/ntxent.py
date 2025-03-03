import torch
import torch.nn.functional as F

from continuity_contrastive_ecg.utils.module import Module


class NTXentLoss(torch.nn.Module, Module):
    def __init__(self, cfg):
        super().__init__()
        self.temperature = cfg.temperature

    def forward(self, z1, z2):
        batch_size = z1.size(0)
        assert batch_size == z2.size(0), "z1 and z2 must have the same batch size"
        assert z1.size(1) == z2.size(1), "z1 and z2 must have the same feature dimension"

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        z = torch.cat([z1, z2], dim=0)  # (2B, D)

        similarity_matrix = torch.mm(z, z.t())  # (2B, 2B)

        logits = similarity_matrix / self.temperature

        labels = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]).to(
            z.device
        )

        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        logits = logits.masked_fill(mask, float("-inf"))

        loss = F.cross_entropy(logits, labels)

        return loss
