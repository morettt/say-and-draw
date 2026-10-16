import torch
from torch import nn
class CausalCodeLM(nn.Module):
    def __init__(self,vocab):
        super().__init__();self.tok=nn.Embedding(vocab,128);self.pos=nn.Embedding(18,128)
        layer=nn.TransformerEncoderLayer(128,4,512,dropout=.1,batch_first=True,activation='gelu',norm_first=True)
        self.blocks=nn.TransformerEncoder(layer,4,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(128);self.head=nn.Linear(128,vocab,bias=False)
    def forward(self,ids):
        t=ids.shape[1];h=self.tok(ids)+self.pos(torch.arange(t,device=ids.device))[None]
        mask=torch.ones(t,t,device=ids.device,dtype=torch.bool).triu(1)
        return self.head(self.norm(self.blocks(h,mask=mask)))
