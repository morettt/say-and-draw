import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval import evaluate_dialogue_regression as evaluation
import run_qwen35_pipeline as pipeline


def judgement(value):
    return evaluation.parse_judgement(json.dumps({
        "scores": dict.fromkeys(evaluation.SCORE_DIMENSIONS, value),
        "reason": "评分依据",
    }))


class SingleModelEvaluationTests(unittest.TestCase):
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
            with patch('sys.argv', argv), patch.object(evaluation, 'generate_answers', return_value=['答一', '答二']) as generate, patch.object(evaluation, 'judge', side_effect=[judgement(5), judgement(3)]) as judge, contextlib.redirect_stdout(io.StringIO()):
                evaluation.main()
            generate.assert_called_once_with(Path('models/qwen3.5-0.8b'), evaluation.load_cases(cases), 160)
            self.assertEqual(judge.call_count, 2)
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(report['summary']['mean_score'], 4)
            self.assertEqual(report['summary']['dimension_mean_scores']['correctness'], 4)
            self.assertEqual(report['evaluation_mode'], 'single_model')
            self.assertEqual(report['rows'][0]['answer'], '答一')
            self.assertEqual(len(report['cases_sha256']), 64)
            self.assertNotIn('baseline_model', report)
            self.assertNotIn('candidate_wins', report['summary'])

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
