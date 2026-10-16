import torch
from torch import nn
class StructuredStrokeDecoder(nn.Module):
 def __init__(self):
  super().__init__()
  self.body=nn.Sequential(nn.Linear(512,768),nn.LayerNorm(768),nn.GELU(),
   nn.Linear(768,512),nn.LayerNorm(512),nn.GELU(),
   nn.Linear(512,512),nn.GELU())
  self.xy=nn.Linear(512,8*32*2)
  self.count=nn.Linear(512,8)
  nn.init.normal_(self.xy.weight,std=.001)
  nn.init.zeros_(self.xy.bias)
  with torch.no_grad():self.xy.bias[0::2]=.5
 def forward(self,q):
  h=self.body(q.reshape(len(q),-1))
  r=self.xy(h).reshape(-1,8,32,2)
  xy=torch.stack((r[:,:,:,0],r[:,:,:,1].sigmoid()),-1)
  return xy,self.count(h)
