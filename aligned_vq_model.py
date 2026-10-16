import json,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from aligned_digit_data import load,BASE

torch.manual_seed(924);torch.set_num_threads(4)
class AlignedVQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv1d(2,64,5,padding=2),nn.GELU(),nn.Conv1d(64,128,4,stride=2,padding=1),nn.GELU(),nn.Conv1d(128,64,4,stride=2,padding=1),nn.GELU(),nn.Conv1d(64,32,3,padding=1))
        self.book=nn.Embedding(256,32);nn.init.uniform_(self.book.weight,-.1,.1)
        self.decoder=nn.Sequential(nn.Conv1d(32,64,3,padding=1),nn.GELU(),nn.ConvTranspose1d(64,128,4,stride=2,padding=1),nn.GELU(),nn.ConvTranspose1d(128,64,4,stride=2,padding=1),nn.GELU(),nn.Conv1d(64,2,5,padding=2),nn.Sigmoid())
    def encode(self,x):
        z=self.encoder(x.transpose(1,2)).transpose(1,2)
        dist=(z.square().sum(-1,keepdim=True)-2*z@self.book.weight.T+self.book.weight.square().sum(-1))
        ids=dist.argmin(-1)
        return z,ids
    def decode(self,ids):
        return self.decoder(self.book(ids).transpose(1,2)).transpose(1,2)
    def forward(self,x):
        z,ids=self.encode(x);q=self.book(ids)
        y=self.decoder((z+(q-z).detach()).transpose(1,2)).transpose(1,2)
        return y,z,q,ids

