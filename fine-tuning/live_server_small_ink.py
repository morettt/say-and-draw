"""Serve the trained digit generator and VQ decoder over HTTP."""
from project_paths import ROOT, SMOOTH_VQ, STRUCTURED_DECODER, CONTROL_GENERATOR

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
sys.path.insert(0, str(ROOT))
from causal_vq_model import CausalCodeLM
from aligned_vq_model import AlignedVQ
from structured_decoder import StructuredStrokeDecoder
retrain_checkpoint = torch.load(CONTROL_GENERATOR, map_location=DEVICE, weights_only=False)
retrain_model = CausalCodeLM(retrain_checkpoint['vocab']).to(DEVICE).eval()
retrain_model.load_state_dict(retrain_checkpoint['model'])
retrain_used = retrain_checkpoint['used'].to(DEVICE)
retrain_vq = AlignedVQ().to(DEVICE).eval()
retrain_vq.load_state_dict(torch.load(SMOOTH_VQ, map_location=DEVICE, weights_only=False)['model'])
retrain_struct = StructuredStrokeDecoder().to(DEVICE).eval()
retrain_struct.load_state_dict(torch.load(STRUCTURED_DECODER, map_location=DEVICE, weights_only=False)['model'])
torch.seed()
@torch.no_grad()
def generate_b(digit, temperature):
    sequence = torch.tensor([[0, 1 + digit]], device='cuda')
    selected = []
    for _ in range(16):
        logits = retrain_model(sequence)[0, -1, 37:]
        values, positions = torch.topk(logits, 2)
        probabilities = torch.softmax(values / temperature, dim=-1)
        code = positions[torch.multinomial(probabilities, 1)[0]]
        selected.append(code)
        sequence = torch.cat([sequence, (code + 37).view(1, 1)], dim=1)
    ids = retrain_used[torch.stack(selected)]
    q = retrain_vq.book(ids[None])
    y, count_logits = retrain_struct(q)
    count = int(count_logits.argmax(-1)[0].item()) + 1
    y = y[0, :count].reshape(count * 32, 2).cpu().numpy()
    points = np.zeros((len(y), 3), dtype=np.float32)
    points[:, :2] = y
    points[::32, 2] = 10
    return ids.cpu().tolist(), points

lock = threading.Lock()


PAGE = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>200条数字模型：实时生成</title>
<style>body{font-family:system-ui;margin:32px auto;max-width:850px;padding:0 16px;background:#f5f5f5;color:#222}
input,button{font-size:22px;padding:9px}input{width:180px}button{cursor:pointer}.box{background:white;padding:20px;border-radius:12px;margin-top:20px}
svg{height:420px;border:1px solid #ddd}.box{overflow-x:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>
<h1>200条数字模型：实时生成</h1>
<p>每次点击都会在云机上重新采样每位数字的16个编号，再由VQ解码器画出笔迹。画布随位数变宽。</p>
<input id="n" value="333" inputmode="numeric"><button id="go">生成</button>
<label>变化程度 <input id="temperature" type="range" min="0.5" max="1.1" step="0.1" value="0.7"><span id="temp_value">0.7</span></label>
<p id="status">等待输入</p><div class="box"><svg id="canvas" viewBox="0 0 400 400"></svg><pre id="codes"></pre></div>
<script>
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function run(){const n=document.getElementById('n').value.trim(),status=document.getElementById('status');
if(!/^[0-9]{1,32}$/.test(n)){status.textContent='请输入1到32位数字';return}
const go=document.getElementById('go'),canvas=document.getElementById('canvas'),codeBox=document.getElementById('codes');
go.disabled=true;canvas.innerHTML='';codeBox.textContent='';status.textContent='正在生成第1位...';const started=performance.now();
const width=Math.max(400,n.length*140);canvas.setAttribute('viewBox','0 0 '+width+' 400');canvas.style.width=width+'px';
const queue=[],lines=[];let ended=false,drawError=null,drawn=0,requestId='';
const drawing=(async()=>{while(!ended||queue.length){if(!queue.length){await pause(12);continue}
const stroke=queue.shift(),line=document.createElementNS('http://www.w3.org/2000/svg','polyline');
line.setAttribute('fill','none');line.setAttribute('stroke','#151515');line.setAttribute('stroke-width','4');
line.setAttribute('stroke-linecap','round');line.setAttribute('stroke-linejoin','round');canvas.appendChild(line);
for(let i=0;i<stroke.length;i++){line.setAttribute('points',stroke.slice(0,i+1).map(p=>(p[0]*width).toFixed(1)+','+(p[1]*400).toFixed(1)).join(' '));await pause(8)}
drawn++;}if(drawError)throw drawError})();
try{const temperature=Number(document.getElementById('temperature').value);
const res=await fetch('/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({number:n,temperature})});
if(!res.ok){const error=await res.json();throw Error(error.error||'推理失败')}
const reader=res.body.getReader(),decoder=new TextDecoder();let buffer='';
while(true){const chunk=await reader.read();if(chunk.done)break;buffer+=decoder.decode(chunk.value,{stream:true});
let cut;while((cut=buffer.indexOf('\\n'))>=0){const row=buffer.slice(0,cut);buffer=buffer.slice(cut+1);if(!row)continue;
const part=JSON.parse(row);requestId=part.request_id;queue.push(...part.strokes);lines.push(n[part.position]+': '+part.codes.join(', '));
codeBox.textContent=lines.join('\\n');status.textContent='模型已生成 '+(part.position+1)+'/'+n.length+' 位，画板正在书写...';}}
ended=true;await drawing;status.textContent='完成｜'+n.length+' 位｜模型逐位流式输出｜画笔逐点播放｜总耗时 '+Math.round(performance.now()-started)+' ms｜请求 #'+requestId;
}catch(e){drawError=e;ended=true;status.textContent='失败：'+e.message}finally{go.disabled=false}}
document.getElementById('go').onclick=run;document.getElementById('n').onkeydown=e=>{if(e.key==='Enter')run()};
document.getElementById('temperature').oninput=e=>{document.getElementById('temp_value').textContent=e.target.value};
</script></html>'''


FLOW_PAGE = PAGE.replace('200\u6761\u6570\u5b57\u6a21\u578b\uff1a\u5b9e\u65f6\u751f\u6210',
                         '\u0056\u0041\u0045\u7f16\u53f7\u6a21\u578b\uff1a\u5b9e\u65f6\u751f\u6210')
FLOW_PAGE = FLOW_PAGE.replace("fetch('/stream'", "fetch('/flow-stream'")
FLOW_PAGE = FLOW_PAGE.replace("lines.push(n[part.position]+': '+part.codes.join(', '));",
                              "lines.push('\u7b2c '+(part.position+1)+' \u4f4d\u5df2\u751f\u6210');")
FLOW_PAGE = FLOW_PAGE.replace('value="0.7"', 'value="0.5"').replace('>0.7</span>', '>0.5</span>')
start = FLOW_PAGE.index('<p>')
end = FLOW_PAGE.index('</p>', start) + 4
FLOW_PAGE = FLOW_PAGE[:start] + '<p>\u8f93\u5165\u6570\u5b57\uff0c\u6a21\u578b\u751f\u6210\u7b14\u753b\u5e76\u9010\u7b14\u64ad\u653e\u3002</p>' + FLOW_PAGE[end:]


FLOW_PAGE = FLOW_PAGE.replace('VAE编号', '自回归 Transformer + VAE')

FLOW_PAGE = FLOW_PAGE.replace("fetch('/flow-stream'", "fetch('/transformer-stream'")

NEW_PAGE = FLOW_PAGE.replace('自回归 Transformer + VAE', '自回归 Transformer + 新解码器').replace("fetch('/transformer-stream'", "fetch('/transformer-qwen-stream'")

NEW_PAGE = NEW_PAGE.replace('value="333" inputmode="numeric"', 'value="N" inputmode="text"')
NEW_PAGE = NEW_PAGE.replace('/^[0-9]{1,32}$/', '/^[0-9N]{1,32}$/')
NEW_PAGE = NEW_PAGE.replace('Math.max(400,n.length*140)', 'n.length*400')
NEW_PAGE = NEW_PAGE.replace('</style>', 'body{max-width:none;margin:0;padding:12px;box-sizing:border-box;width:100vw;height:100vh;overflow:hidden}h1{margin:0 0 4px}p{margin:4px 0}.box{position:relative;background:transparent;padding:0;margin-top:8px;border-radius:0;overflow-x:auto;overflow-y:hidden}svg{display:block;margin:0;border:0}pre{position:absolute;right:8px;bottom:4px;max-width:35vw;max-height:50px;overflow:auto;margin:0;font-size:11px;background:#ffffffbb}</style>')
NEW_PAGE = NEW_PAGE.replace("const width=n.length*400;canvas.setAttribute('viewBox','0 0 '+width+' 400');canvas.style.width=width+'px';", "const inkHeight=220,inkWidth=n.length*inkHeight,width=Math.max(window.innerWidth-24,inkWidth),top=canvas.getBoundingClientRect().top,height=Math.max(260,window.innerHeight-top-12),shiftX=(width-inkWidth)/2,shiftY=(height-inkHeight)/2;canvas.setAttribute('viewBox','0 0 '+width+' '+height);canvas.style.width=width+'px';canvas.style.height=height+'px';")
NEW_PAGE = NEW_PAGE.replace("(p[0]*width).toFixed(1)+','+(p[1]*400).toFixed(1)", "(p[0]*inkWidth+shiftX).toFixed(1)+','+(p[1]*inkHeight+shiftY).toFixed(1)")
# Restore the collection canvas geometry after digit x was expanded by 1/.2 in prepare.py.
NEW_PAGE = NEW_PAGE.replace('const stroke=queue.shift()', 'const job=queue.shift(),stroke=job.stroke')
NEW_PAGE = NEW_PAGE.replace('queue.push(...part.strokes)', 'queue.push(...part.strokes.map(stroke=>({stroke,position:part.position})))')
NEW_PAGE = NEW_PAGE.replace('(p[0]*inkWidth+shiftX).toFixed(1)', '(shiftX+inkHeight*(job.position+.5+(p[0]*n.length-job.position-.5)*(n[job.position]===\'N\'?2.15:.43))).toFixed(1)')
FIXED_PAGE = NEW_PAGE.replace("fetch('/transformer-qwen-stream'", "fetch('/transformer-fixed-stream'").replace("(n[job.position]==='N'?2.15:.43)", '.43').replace('自回归 Transformer + 新解码器', 'N 坐标统一实验')
B_PAGE = FIXED_PAGE.replace("fetch('/transformer-fixed-stream'", "fetch('/transformer-qwen-stream'").replace('value="N"', 'value="U"').replace('/^[0-9N]{1,32}$/', '/^[0-9A-Z]{1,32}$/').replace('N 坐标统一实验', 'B 字母实验')
Q_PAGE = B_PAGE.replace("/transformer-new-stream","/transformer-qwen-stream").replace('B 字母实验','Qwen3.5 第一阶段')


B_PAGE = B_PAGE.replace('B \u5b57\u6bcd\u5b9e\u9a8c', 'A+B \u5b57\u6bcd\u5b9e\u9a8c')

B_PAGE = B_PAGE.replace('A+B ', 'A-D ')

B_PAGE = B_PAGE.replace('A-D ', 'A-F ')

B_PAGE = B_PAGE.replace('A-F ', 'A-K ')

B_PAGE = B_PAGE.replace('A-K ', 'A-Z ')
B_PAGE = B_PAGE.replace('/^[0-9A-Z]{1,32}$/','/^[0-9A-Z]+$/').replace('请输入1到32位数字','请输入数字或大写字母').replace('输入数字，','输入数字或大写字母，')
B_PAGE = B_PAGE.replace('const inkHeight=220,inkWidth=n.length*inkHeight', 'const inkHeight=220,slotWidth=145,inkWidth=(n.length-1)*slotWidth+inkHeight')
B_PAGE = B_PAGE.replace('(shiftX+inkHeight*(job.position+.5+', '(shiftX+slotWidth*job.position+inkHeight*(.5+')

Q_PAGE = B_PAGE.replace('A-Z 字母实验', 'Qwen3.5 笔迹生成').replace('</h1>', '</h1><p>当前模型：Qwen3.5 · FP32 更新续训版 · 笔迹评分 89.38 · BF16 推理</p>')
SMALL_PAGE = B_PAGE.replace("fetch('/transformer-qwen-stream'", "fetch('/transformer-small-stream'")
SMALL_PAGE = SMALL_PAGE.replace('</h1>', '</h1><p>独立复训模型：2026-10-15</p>')
STROKE_PAGE = FLOW_PAGE.replace('自回归 Transformer + VAE', '自回归 Transformer + 自动抬笔').replace("fetch('/transformer-stream'", "fetch('/transformer-strokes-stream'")

def smooth_points(points):
    result = points.copy()
    for i in range(1, len(points) - 1):
        if points[i, 2] <= 0 and points[i + 1, 2] <= 0:
            result[i, :2] = (.25 * points[i - 1, :2]
                             + .5 * points[i, :2]
                             + .25 * points[i + 1, :2])
    return result

request_count = 0


@torch.no_grad()
def sample_digit(digit, temperature):
    seq = prompt_digit(digit)
    codes = []
    for _ in range(16):
        logits = model(torch.tensor(seq, device='cuda')[None])[0, -1, CODE0:NEXT]
        top_logits, top_ids = torch.topk(logits, 8)
        probs = torch.softmax(top_logits / temperature, dim=-1)
        token = int(top_ids[torch.multinomial(probs, 1)].item()) + CODE0
        seq.append(token)
        codes.append(token - CODE0)
    return codes


@torch.no_grad()
def infer(number, temperature):
    global request_count
    start = time.perf_counter()
    digits = str(number)
    if len(digits) <= 3:
        slots = {1: [.46], 2: [.39, .59], 3: [.31, .50, .69]}[len(digits)]
        digit_scale = .2
    else:
        slots = [(i + .5) / len(digits) for i in range(len(digits))]
        digit_scale = .6 / len(digits)
    canvas_width = max(400, len(digits) * 140)
    strokes, all_codes = [], []
    with lock:
        for pos, char in enumerate(digits):
            ids = sample_digit(int(char), temperature)
            all_codes.append(ids)
            points = vq.decode(torch.tensor(ids, device='cuda')[None])[0].cpu().numpy()
            points[0, 2] = 10
            if not b:
                points = smooth_points(points)
            current = []
            for i, p in enumerate(points):
                if i and p[2] > 0 and current:
                    strokes.append(current)
                    current = []
                current.append([
                    float(np.clip((p[0] - .5) * digit_scale + slots[pos], 0, 1)),
                    float(np.clip(p[1], 0, 1)),
                ])
            if current:
                strokes.append(current)
        torch.cuda.synchronize()
        request_count += 1
        count = request_count
    return {'number': number, 'codes': all_codes, 'strokes': strokes,
            'canvas_width': canvas_width,
            'temperature': temperature,
            'inference_ms': round((time.perf_counter() - start) * 1000),
            'request_id': count}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ('/flow', '/transformer'):
            body = FLOW_PAGE.encode('utf-8')
        elif self.path == '/transformer-qwen':
            body = (ROOT/'qwen_chat_page.html').read_bytes()
        elif self.path == '/transformer-new':
            body = NEW_PAGE.encode('utf-8')
        elif self.path == '/new':
            body = FIXED_PAGE.encode('utf-8')
        elif self.path == '/transformer-strokes':
            body = STROKE_PAGE.encode('utf-8')
        elif self.path == '/compare':
            body = Path('/root/autodl-tmp/digit_more_data/残差分词器对比.html').read_bytes()
        elif self.path == '/generate-compare':
            body = Path('/root/autodl-tmp/digit_more_data/生成效果对比.html').read_bytes()
        elif self.path in ('/', '/index.html', '/transformer-small'):
            body = SMALL_PAGE.encode('utf-8')
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path not in ('/chat', '/generate', '/stream', '/flow-stream', '/transformer-stream', '/transformer-new-stream', '/transformer-fixed-stream', '/transformer-strokes-stream', '/transformer-qwen-stream', '/transformer-small-stream'):
            self.send_error(404)
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if size > 1000000:
                raise ValueError('请求过大')
            payload = json.loads(self.rfile.read(size))
            value = payload['number']
            if self.path == '/chat':
                if not isinstance(value,str) or not value.strip(): raise ValueError('输入不能为空')
                with lock:
                    result=infer_qwen_chat(value,float(payload.get('temperature',.7)),q_model,q_tok,q_code_ids,q_code_map,retrain_vq,retrain_struct,history=payload.get('history',[]))
                body=json.dumps(result,ensure_ascii=False).encode('utf-8')
                self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8');self.end_headers();self.wfile.write(body);return
            if self.path == '/transformer-qwen-stream' and not (isinstance(value,str) and value.isascii() and value.isalnum()):
                import re
                found=re.findall(r'(?<![A-Za-z])[A-Z0-9](?![A-Za-z])',str(value),re.I)
                if not found: raise ValueError('请在指令中写出要画的数字或大写字母')
                value=found[-1].upper()
            if not isinstance(value, str) or not value.isascii() or not (len(value) >= 1) or not all(c.isdigit() or (self.path in ('/transformer-new-stream', '/transformer-fixed-stream') and c == 'N') or (self.path in ('/transformer-new-stream','/transformer-qwen-stream','/transformer-small-stream') and c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ') for c in value):
                raise ValueError('请输入数字或大写字母')
            temperature = float(payload.get('temperature', .7))
            if not (.5 <= temperature <= 1.1):
                raise ValueError('变化程度超出范围')
            if self.path in ('/stream', '/flow-stream', '/transformer-stream', '/transformer-new-stream', '/transformer-fixed-stream', '/transformer-strokes-stream', '/transformer-qwen-stream', '/transformer-small-stream'):
                self.stream(value, temperature, flow=self.path in ('/flow-stream','/transformer-stream'), new_decoder=self.path in ('/transformer-new-stream','/transformer-fixed-stream','/transformer-small-stream'), use_strokes=self.path == '/transformer-strokes-stream', fixed=self.path == '/transformer-fixed-stream', b=self.path == '/transformer-small-stream', qwen=self.path in ('/transformer-qwen-stream','/transformer-new-stream'))
                return
            output, status = infer(value, temperature), 200
        except Exception as exc:
            output, status = {'error': str(exc)}, 400
        body = json.dumps(output, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def stream(self, digits, temperature, flow=False, new_decoder=False, use_strokes=False, fixed=False, b=False, qwen=False):
        global request_count
        if len(digits) <= 3:
            slots = {1: [.46], 2: [.39, .59], 3: [.31, .50, .69]}[len(digits)]
            digit_scale = .2
        else:
            slots = [(i + .5) / len(digits) for i in range(len(digits))]
            digit_scale = .6 / len(digits)
        with lock:
            request_count += 1
            count = request_count
        self.send_response(200)
        self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        started = time.perf_counter()
        for position, char in enumerate(digits):
            with lock, torch.no_grad():
                if use_strokes:
                    codes, points = generate_stroke(stroke_model, stroke_vq, stroke_used, int(char), temperature)
                elif qwen:
                    codes, points = generate_qwen(char, temperature)
                elif b:
                    codes, points = generate_b(int(char) if char.isdigit() else {'N':10,'B':11,'A':12,'C':13,'D':14,'E':15,'F':16,'G':17,'H':18,'I':19,'J':20,'K':21,'L':22,'M':23,'O':24,'P':25,'Q':26,'R':27,'S':28,'T':29,'U':30,'V':31,'W':32,'X':33,'Y':34,'Z':35}[char], temperature)
                elif flow:
                    codes, points = generate_causal_vq(b_model if b else (fixed_model if fixed else causal_model), b_vq if b else (fixed_vq if fixed else causal_vq), b_used if b else (fixed_used if fixed else causal_used),
                                                       (int(char) if char.isdigit() else {'N':10,'B':11,'A':12,'C':13,'D':14,'E':15,'F':16,'G':17,'H':18,'I':19,'J':20,'K':21,'L':22,'M':23,'O':24,'P':25,'Q':26,'R':27,'S':28,'T':29,'U':30,'V':31,'W':32,'X':33,'Y':34,'Z':35}[char]), temperature, decoder=(b_decoder if b else (fixed_decoder if fixed else residual_decoder)) if new_decoder else None)
                else:
                    codes = sample_digit(int(char), temperature)
                    points = vq.decode(torch.tensor(codes, device='cuda')[None])[0].cpu().numpy()
            points[0, 2] = 10
            if not b:
                points = smooth_points(points)
            strokes, current = [], []
            for i, p in enumerate(points):
                if i and p[2] > 0 and current:
                    strokes.append(current)
                    current = []
                current.append([
                    float((position + p[0]) / len(digits)) if new_decoder else float(np.clip((p[0] - .5) * (1.2 if char == 'N' else digit_scale) + slots[position], 0, 1)),
                    float(np.clip(p[1], 0, 1)),
                ])
            if current:
                strokes.append(current)
            event = {'request_id': count, 'position': position, 'digit': char,
                     'codes': codes, 'strokes': strokes,
                     'model_elapsed_ms': round((time.perf_counter() - started) * 1000)}
            self.wfile.write((json.dumps(event, ensure_ascii=False) + '\n').encode('utf-8'))
            self.wfile.flush()


if __name__ == '__main__':
    server = ThreadingHTTPServer(('0.0.0.0', int(os.getenv('PORT', '6006'))), Handler)
    print('ready http://127.0.0.1:6006', flush=True)
    server.serve_forever()






