import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

from continuity_contrastive_ecg.utils.module import Module


class Wav2Vec4ECG(nn.Module, Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        wav2vec2_config = Wav2Vec2Config(**cfg.encoder)
        self.encoder = Wav2Vec2Model(wav2vec2_config)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        h = self.encoder(x).last_hidden_state
        h_pooled = self.global_pool(h.transpose(1, 2)).transpose(1, 2)
        return h_pooled
