# SayAndDraw · 说画

输入文字，模型用普通文字回复，或逐笔绘制手写内容。这里开放推理代码和网页界面。**当前权重只训练了单个数字 0～9**；字母、汉字和其他画面是项目的后续目标，当前版本还不能可靠生成。

## 运行原理

画图请求进入 Qwen3.5-0.8B 后，模型先回复一句文字，再生成 `<DRAW>` 和 16 个 `<M000>`～`<M255>` 编号。动作码本和笔画解码器将编号还原成 64 个二维点，网页按顺序画出线条。普通聊天请求直接输出文字。Qwen、码本和解码器的权重必须配套使用。

## 安装

建议 Python 3.12、NVIDIA GPU 和 CUDA 12.4。下面是本项目验证过的 PyTorch 版本；其他环境可能需要换成对应的 PyTorch 安装命令。

```bash
#部署环境
conda create -n saydray python=3.12 -y
conda activate saydray
#下载依赖
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

CPU 也能尝试运行，速度会慢很多。此时可直接安装 `requirements.txt`，启动时加 `--device cpu`。

## 放置模型文件

模型权重已发布在 [魔塔 SayAndDraw-v0.1](https://www.modelscope.cn/models/morelle/SayAndDraw-v0.1)。本代码仓库不包含权重。下载整个模型仓库后，将文件放入下列结构，或用 `--model-dir` 指向下载目录：

```text
models/
├── llm/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── chat_template.jinja
│   └── generation_config.json
├── aligned_vq_smooth.pt
└── residual_decoder.pt
```

`llm/` 是**从第一阶段权重继续训练统一输出格式**得到的第二阶段 Qwen 模型，另外两个 `.pt` 文件分别是配套码本和笔画解码器。这三个权重文件需配套使用。

## 启动网页

```bash
python app.py --model-dir models --port 6006
```

打开 <http://127.0.0.1:6006/>，可以输入“请你画一个数字3”，也可以发送普通聊天消息。服务默认只监听本机；要让局域网设备访问，可加 `--host 0.0.0.0`。

接口是 `POST /api/generate`，请求示例：

```json
{"messages":[{"role":"user","content":"请你画一个数字3"}],"temperature":0.7}
```

画图回复含 `reply`、`tokens`、`strokes` 和 `elapsed_ms`；聊天回复的 `strokes` 为 `null`。`temperature: 0` 使用确定性选码，较高温度会增加变化，也可能降低画图质量。

## 当前限制

- 模型目前只学习单个数字 0～9，不能据此宣称可以画任意内容。
- 现有笔迹数据未单独编码抬笔位置。数字 4 和 5 使用预览版的固定抬笔位置；后续扩展到字母和多笔画图形时，需要训练显式的抬笔表示。
- 训练时的逐编号准确率只衡量编号预测，不能代替肉眼观察绘制质量。
- 本仓库只提供推理与操作界面，不包含训练数据和训练脚本。

代码采用仓库中的 MIT License。Qwen 基础模型及已发布权重遵守各自的模型许可。


