import contextlib
from collections import UserDict
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from eval import evaluate_dialogue_regression as evaluation
import run_qwen35_pipeline as pipeline


def judgement(value):
    return evaluation.parse_judgement(json.dumps({
        "scores": dict.fromkeys(evaluation.SCORE_DIMENSIONS, value),
        "reason": "评分依据",
    }))


class SingleModelEvaluationTests(unittest.TestCase):
    def test_generation_accepts_mapping_tokenizer_outputs(self):
        for rendered in ([11, 12, 13], {'input_ids': [11, 12, 13]}, UserDict({'input_ids': [11, 12, 13]})):
            with self.subTest(return_type=type(rendered).__name__):
                tokenizer = MagicMock()
                tokenizer.apply_chat_template.return_value = rendered
                tokenizer.decode.return_value = '回答'
                tokenizer.pad_token_id = 0
                model = MagicMock()
                model.to.return_value = model
                model.eval.return_value = model
                generated = MagicMock()
                model.generate.return_value = generated
                torch = SimpleNamespace(
                    cuda=SimpleNamespace(is_available=lambda: False),
                    float32='float32', long='long', tensor=MagicMock(),
                    inference_mode=contextlib.nullcontext,
                )
                transformers = SimpleNamespace(
                    AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
                    Qwen3_5ForConditionalGeneration=SimpleNamespace(from_pretrained=lambda *a, **k: model),
                    TextStreamer=MagicMock(),
                )
                console = io.StringIO()
                def generate_live(**kwargs):
                    # Verify the question is visible before model.generate runs.
                    self.assertIn('[1/1]', console.getvalue())
                    self.assertIn('用户：你好', console.getvalue())
                    self.assertIn('模型回答（生成中）：', console.getvalue())
                    self.assertIs(kwargs['streamer'], transformers.TextStreamer.return_value)
                    print('正在', end='', flush=True)
                    self.assertIn('正在', console.getvalue())
                    print('回答', flush=True)
                    self.assertNotIn('[生成完成', console.getvalue())
                    return generated
                model.generate.side_effect = generate_live
                with patch.dict('sys.modules', {'torch': torch, 'transformers': transformers}), contextlib.redirect_stdout(console):
                    answers = evaluation.generate_answers(Path('model'), [{'messages': [{'role': 'user', 'content': '你好'}]}], 160)
                self.assertEqual(answers, ['回答'])
                torch.tensor.assert_called_once_with([[11, 12, 13]], dtype='long', device='cpu')
                generated.__getitem__.assert_called_once_with((0, slice(3, None)))
                self.assertEqual(model.generate.call_args.kwargs['pad_token_id'], 0)
                transformers.TextStreamer.assert_called_once_with(tokenizer, skip_prompt=True, skip_special_tokens=False)
                self.assertIn('[生成完成 1/1', console.getvalue())

    def test_output_directory_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = ['evaluate', '--model', 'model', '--label', 'base', '--output', directory]
            error = io.StringIO()
            with patch('sys.argv', argv), patch.object(evaluation, 'generate_answers') as generate, contextlib.redirect_stderr(error):
                with self.assertRaises(SystemExit) as stopped:
                    evaluation.main()
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn('--output 必须指定报告文件', error.getvalue())
            generate.assert_not_called()

    def test_parse_scores_and_reject_invalid_judgements(self):
        result = evaluation.parse_judgement('```json\n' + json.dumps({
            "scores": dict(zip(evaluation.SCORE_DIMENSIONS, [5, 4, 3, 2])),
            "reason": "有遗漏",
        }) + '\n```')
        self.assertEqual(result['score'], 3.5)
        for invalid in ({}, {"winner": "A", "a_score": 5, "b_score": 4},
                        {"scores": dict.fromkeys(evaluation.SCORE_DIMENSIONS, 6), "reason": "错误"},
                        {"scores": dict.fromkeys(evaluation.SCORE_DIMENSIONS, True), "reason": "错误"},
                        {"scores": dict.fromkeys(evaluation.SCORE_DIMENSIONS, 5), "reason": ""}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                evaluation.parse_judgement(json.dumps(invalid))

    def test_single_answer_api_payload(self):
        response = io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps({
            "scores": dict.fromkeys(evaluation.SCORE_DIMENSIONS, 4), "reason": "基本准确",
        })}}]}).encode())
        messages = [{"role": "user", "content": "你好"}]
        with patch.object(evaluation.urllib.request, 'urlopen', return_value=response) as call:
            result = evaluation.judge('https://example.test/v1', 'test-key', 'judge', messages, '你好！')
        payload = json.loads(call.call_args.args[0].data)
        self.assertIn('你好！', payload['messages'][1]['content'])
        self.assertNotIn('ANSWER_B', payload['messages'][1]['content'])
        self.assertEqual(payload['temperature'], 0)
        self.assertIs(payload['enable_thinking'], False)
        self.assertEqual(result['score'], 4)

    def test_main_loads_one_model_and_writes_independent_report(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = Path(directory)/'cases.jsonl'
            output = Path(directory)/'reports'/'base.json'
            cases.write_text('\n'.join(json.dumps({
                "id": str(i), "messages": [{"role": "user", "content": "你好"}],
            }) for i in range(2)), encoding='utf-8')
            argv = ['evaluate', '--model', 'models/qwen3.5-0.8b', '--label', 'base',
                    '--cases', str(cases), '--output', str(output)]
            console = io.StringIO()
            def score(*args):
                self.assertIn(f'[评分 {judge.call_count}/2]', console.getvalue())
                return judgement(5 if judge.call_count == 1 else 3)
            with patch('sys.argv', argv), patch.object(evaluation, 'generate_answers', return_value=['答一', '答二']) as generate, patch.object(evaluation, 'judge', side_effect=score) as judge, contextlib.redirect_stdout(console):
                evaluation.main()
            generate.assert_called_once_with(Path('models/qwen3.5-0.8b'), evaluation.load_cases(cases), 160)
            self.assertEqual(judge.call_count, 2)
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(report['summary']['mean_score'], 4)
            self.assertEqual(report['summary']['dimension_mean_scores']['correctness'], 4)
            self.assertEqual(report['evaluation_mode'], 'single_model')
            self.assertEqual(report['judge_model'], 'qwen3.8-flash')
            self.assertIs(report['judge_enable_thinking'], False)
            self.assertEqual(report['rows'][0]['answer'], '答一')
            self.assertEqual(len(report['cases_sha256']), 64)
            self.assertNotIn('baseline_model', report)
            self.assertNotIn('candidate_wins', report['summary'])
            self.assertIn('裁判评语：评分依据', console.getvalue())
            self.assertIn('评估完成，平均分：4.00/5', console.getvalue())

    def test_empty_cases_fail_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = Path(directory)/'empty.jsonl'
            cases.write_text('', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '不能为空'):
                evaluation.load_cases(cases)

    def test_pipeline_passes_only_current_model(self):
        model = pipeline.STAGES[5]['model']
        with patch.object(pipeline.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            run.return_value.returncode = 0
            pipeline.run_dialogue_eval('stage5', model, False)
        command = run.call_args.args[0]
        self.assertEqual(Path(command[2]), Path('eval/evaluate_dialogue_regression.py'))
        self.assertEqual(command[command.index('--model')+1], str(model))
        self.assertNotIn('--baseline', command)
        self.assertNotIn('--candidate', command)


if __name__ == '__main__':
    unittest.main()
