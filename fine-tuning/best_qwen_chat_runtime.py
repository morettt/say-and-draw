import torch

@torch.inference_mode()
def infer(prompt,temperature,model,tok,code_ids,code_map,vq,decoder,history=None):
    system='You are a helpful assistant. Reply normally. For drawing, say a short sentence and then emit <DRAW>, 16 motion codes, <END_DRAW>. Do not emit <CHAT>.'
    messages=[{'role':'system','content':system}]
    for row in (history or [])[-12:]:
        if isinstance(row,dict) and row.get('role') in ('user','assistant') and isinstance(row.get('content'),str):
            messages.append({'role':row['role'],'content':row['content'][:4000]})
    messages.append({'role':'user','content':prompt})
    pre=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)['input_ids']
    ids=torch.tensor([pre],device='cuda');generated=[];codes=[];drawing=False
    draw=tok.convert_tokens_to_ids('<DRAW>');end=tok.convert_tokens_to_ids('<END_DRAW>')
    for _ in range(256):
        logits=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[0,-1].float()
        if drawing:
            vals,ix=logits[code_ids].topk(2);n=int(code_ids[ix[torch.multinomial((vals/max(float(temperature),.05)).softmax(-1),1)[0]]])
        else:
            n=int(logits.argmax())
        generated.append(n);ids=torch.cat([ids,torch.tensor([[n]],device='cuda')],1)
        if n==draw:drawing=True
        elif n in code_map and drawing:
            codes.append(code_map[n])
        if len(codes)==16:generated.append(end);break
        if n==tok.eos_token_id:break
    strokes=[]
    if len(codes)==16:
        xy,count=decoder(vq.book(torch.tensor(codes,device='cuda')[None]));strokes=xy[0,:int(count.argmax(-1)[0])+1].cpu().tolist()
    text=[]
    for n in generated:
        if n==draw or n in code_map:break
        text.append(n)
    return {'reply':tok.decode(text,skip_special_tokens=True).strip(),'codes':codes,'strokes':strokes,'history_content':tok.decode([n for n in generated if n!=tok.eos_token_id],skip_special_tokens=False)}
