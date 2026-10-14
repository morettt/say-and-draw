"""SayAndDraw: chat or generate handwritten strokes from a text request."""

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from models import AlignedVQ, ResidualDecoder


SYSTEM_PROMPT = (
    "You are a helpful assistant. Reply normally. For drawing, say a short "
    "sentence and then emit <DRAW>, 16 motion codes, <END_DRAW>."
)
WEB_PAGE = Path(__file__).resolve().parent / "web" / "index.html"


class DrawingModel:
    def __init__(self, model_dir: Path, device: str):
        llm_dir = model_dir / "llm"
        vq_file = model_dir / "aligned_vq_smooth.pt"
        decoder_file = model_dir / "residual_decoder.pt"
        for path in (llm_dir / "config.json", vq_file, decoder_file):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing model file: {path}. Run python download_model.py first."
                )

        self.device = torch.device(device)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(llm_dir)
        self.llm = Qwen3_5ForConditionalGeneration.from_pretrained(
            llm_dir, dtype=dtype, attn_implementation="eager"
        ).to(self.device).eval()
        self.llm.config.use_cache = True

        self.vq = AlignedVQ().to(self.device).eval()
        self.vq.load_state_dict(
            torch.load(vq_file, map_location=self.device, weights_only=True)["model"]
        )
        self.decoder = ResidualDecoder().to(self.device).eval()
        self.decoder.load_state_dict(
            torch.load(decoder_file, map_location=self.device, weights_only=True)["decoder"]
        )

        self.draw_id = self._token_id("<DRAW>")
        self.end_draw_id = self._token_id("<END_DRAW>")
        self.code_id_list = [self._token_id(f"<M{i:03d}>") for i in range(256)]
        self.code_ids = torch.tensor(self.code_id_list, device=self.device)
        self.forbidden_text_ids = torch.tensor(
            [self.end_draw_id, *self.code_id_list], device=self.device
        )
        self.stop_ids = {
            token_id
            for token_id in (
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            )
            if token_id is not None
        }
        self.lock = threading.Lock()

    def _token_id(self, token: str) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"The model tokenizer is missing {token}")
        return token_id

    def _prompt(self, messages: list[dict]) -> torch.Tensor:
        chat = [{"role": "system", "content": SYSTEM_PROMPT}]
        chat.extend(messages[-8:])
        encoded = self.tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        token_ids = encoded["input_ids"]
        return torch.tensor([token_ids], device=self.device)

    @staticmethod
    def _select(logits: torch.Tensor, temperature: float, top_k: int) -> int:
        values, indices = torch.topk(logits, top_k)
        if temperature == 0:
            return int(indices[0])
        probabilities = torch.softmax(values / max(0.2, temperature), dim=0)
        return int(indices[torch.multinomial(probabilities, 1)])

    @torch.inference_mode()
    def generate(self, messages: list[dict], temperature: float) -> dict:
        started = time.perf_counter()
        ids = self._prompt(messages)
        text_ids = []
        drawing = False

        for _ in range(100):
            logits = self.llm(input_ids=ids).logits[0, -1].float()
            logits[self.forbidden_text_ids] = -float("inf")
            next_id = self._select(logits, temperature, 40)
            if next_id == self.draw_id:
                drawing = True
                break
            if next_id in self.stop_ids:
                break
            text_ids.append(next_id)
            ids = torch.cat((ids, ids.new_tensor([[next_id]])), dim=1)

        reply = self.tokenizer.decode(text_ids, skip_special_tokens=True).strip()
        if not drawing:
            return {
                "reply": reply,
                "strokes": None,
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            }

        ids = torch.cat((ids, ids.new_tensor([[self.draw_id]])), dim=1)
        codes = []
        for _ in range(16):
            logits = self.llm(input_ids=ids).logits[0, -1, self.code_ids].float()
            code = self._select(logits, temperature, 2)
            codes.append(code)
            ids = torch.cat((ids, self.code_ids[code].view(1, 1)), dim=1)

        quantized = self.vq.book(torch.tensor([codes], device=self.device))
        points = self.decoder(quantized)[0].float().cpu().numpy()
        points = np.clip(points, 0.0, 1.0)

        # The current digit dataset does not encode pen-up events. Keep the
        # original 4/5 digit split used in the validated preview application.
        match = re.search(r"[0-9]", messages[-1]["content"])
        digit = int(match.group()) if match else None
        pen_lifts = {4: 32, 5: 54}
        lift = pen_lifts.get(digit)
        strokes = []
        current = []
        for index, point in enumerate(points):
            if index == lift and current:
                strokes.append(current)
                current = []
            current.append([float(point[0]), float(point[1])])
        if current:
            strokes.append(current)

        return {
            "reply": reply,
            "tokens": [f"<M{code:03d}>" for code in codes],
            "strokes": strokes,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }


def serve(model: DrawingModel, host: str, port: int):
    page = WEB_PAGE.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            self._respond(200, page, "text/html; charset=utf-8")

        def do_POST(self):
            if self.path != "/api/generate":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 1 <= size <= 20000:
                    raise ValueError("请求内容长度无效")
                data = json.loads(self.rfile.read(size))
                messages = data["messages"]
                if (
                    not isinstance(messages, list)
                    or not 1 <= len(messages) <= 16
                    or any(
                        not isinstance(item, dict)
                        or item.get("role") not in ("user", "assistant")
                        or not isinstance(item.get("content"), str)
                        or len(item["content"]) > 1000
                        for item in messages
                    )
                    or messages[-1]["role"] != "user"
                ):
                    raise ValueError("消息格式无效")
                temperature = float(data.get("temperature", 0.7))
                if not 0 <= temperature <= 2:
                    raise ValueError("temperature 必须在 0 到 2 之间")
                with model.lock:
                    result = model.generate(messages, temperature)
                self._respond(200, json.dumps(result, ensure_ascii=False).encode(), "application/json; charset=utf-8")
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self._respond(400, json.dumps({"error": str(exc)}, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _respond(self, status: int, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"SayAndDraw ready at http://{host}:{port}", flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Run the SayAndDraw web interface")
    parser.add_argument(
        "--model-dir", type=Path, default=Path(__file__).resolve().parent / "models"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu or install CUDA PyTorch")
    model = DrawingModel(args.model_dir.resolve(), device)
    serve(model, args.host, args.port)


if __name__ == "__main__":
    main()
