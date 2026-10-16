import torch
from causal_vq_model import CausalCodeLM
class ConditionedCodeLM(CausalCodeLM):
 def forward(self,ids):
  t=ids.shape[1]
  condition=self.tok(ids[:,1])[:,None,:]
  h=self.tok(ids)+self.pos(torch.arange(t,device=ids.device))[None]+.5*condition
  mask=torch.ones(t,t,device=ids.device,dtype=torch.bool).triu(1)
  return self.head(self.norm(self.blocks(h,mask=mask)))
