"""Architectures needed to decode the 16 drawing tokens into pen coordinates."""

import torch
from torch import nn


class AlignedVQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 64, 5, padding=2), nn.GELU(),
            nn.Conv1d(64, 128, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(128, 64, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(64, 32, 3, padding=1),
        )
        self.book = nn.Embedding(256, 32)
        self.decoder = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1), nn.GELU(),
            nn.ConvTranspose1d(64, 128, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(64, 2, 5, padding=2), nn.Sigmoid(),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x) * 0.5


class ResidualDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv1d(32, 96, 3, padding=1)
        self.low = nn.Sequential(
            ResidualBlock(96, 1), ResidualBlock(96, 2), ResidualBlock(96, 4)
        )
        self.mid = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(96, 96, 3, padding=1),
            ResidualBlock(96, 1), ResidualBlock(96, 2),
        )
        self.high = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(96, 96, 3, padding=1),
            ResidualBlock(96, 1), ResidualBlock(96, 2), ResidualBlock(96, 4),
        )
        self.output = nn.Sequential(
            nn.Conv1d(96, 48, 3, padding=1), nn.GELU(),
            nn.Conv1d(48, 2, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, q):
        x = self.input(q.transpose(1, 2))
        return self.output(self.high(self.mid(self.low(x)))).transpose(1, 2)
