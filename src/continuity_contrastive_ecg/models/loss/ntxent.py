import torch

from continuity_contrastive_ecg.utils.module import Module


class NTXentLoss(torch.nn.Module, Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = torch.nn.CosineSimilarity()

    def forward(self, z1, z2):
        raise NotImplementedError
