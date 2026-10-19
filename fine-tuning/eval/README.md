# 评估工具

以下命令在 `fine-tuning/` 目录执行。

## 单模型对话评分

`evaluate_dialogue_regression.py` 每次加载一份 Qwen 模型，由独立裁判逐题评分。
默认用例为本目录的 `dialogue_holdout.jsonl`，包含 20 条未参与训练的问题。

评估未微调的基础模型：

```powershell
python eval/evaluate_dialogue_regression.py --model models/qwen3.5-0.8b --label base_qwen35 --output eval/dialogue_runs/base_qwen35.json
```

评估微调后的模型：

```powershell
python eval/evaluate_dialogue_regression.py --model runs/qwen_confirmation/best --label stage5 --output eval/dialogue_runs/stage5.json
```

评估第一阶段时，`--model runs/qwen_stage1_oldmethod_restart` 会读取最近完成的一次训练；也可指定其中某次 `run_.../` 的完整路径。

默认裁判为 `qwen3.8-flash`，请求显式关闭思考模式。可通过 `ALI_QWEN_API_KEY` 配置裁判密钥，`--cases` 指定其他测试用例，`--judge-model` 指定裁判模型。
终端仅输出每题一行 `{"id": "chat_001", "score": 3.5}` 和最后的 JSON 汇总，不打印问题、模型回答、耗时或加载进度。问题、回答、详细的四项分数和评语保存在报告中。比较报告时保持测试用例、裁判和生成参数一致。
Qwen 训练流水线会在每阶段结束后调用此脚本；`--skip-dialogue-eval` 可跳过。

## 笔迹质量评估

```powershell
python eval/evaluate_retrain.py
```

需要 CUDA 和已训练的小模型、VQ、结构化笔画解码器权重。
默认评估 `runs/small_transformer/control_gen_1600.pt`，可通过 `GENERATOR` 环境变量指定其他生成器。
评估结果和缓存写入 `runs/small_transformer/eval/`。

两份脚本也支持模块形式，例如 `python -m eval.evaluate_dialogue_regression --help`。
