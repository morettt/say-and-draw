import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import qwen_model_setup as setup
from project_paths import ROOT, QWEN_MODEL_ID
import run_qwen35_pipeline as pipeline


class ModelSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model_dir = Path(self.temp.name)/'models'/'qwen3.5-0.8b'
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def write_model(self, sharded=False):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
            (self.model_dir/name).write_text('{}')
        if sharded:
            (self.model_dir/'model.safetensors.index.json').write_text(json.dumps({
                'weight_map': {'a':'model-00001.safetensors', 'b':'model-00002.safetensors'},
            }))
            for name in ('model-00001.safetensors', 'model-00002.safetensors'):
                (self.model_dir/name).write_bytes(b'weights')
        else:
            (self.model_dir/'model.safetensors').write_bytes(b'weights')

    def test_missing_model_downloaded_before_return(self):
        def download(command, **kwargs):
            self.assertEqual(command, ['modelscope', 'download', '--model', QWEN_MODEL_ID,
                                       '--local_dir', str(self.model_dir)])
            self.assertEqual(kwargs['cwd'], ROOT)
            self.write_model()
            from subprocess import CompletedProcess
            return CompletedProcess(command, 0)
        with patch.object(setup, 'modelscope_executable', return_value='modelscope'), patch.object(setup.subprocess, 'run', side_effect=download) as run:
            self.assertEqual(setup.ensure_qwen_model(model_dir=self.model_dir), self.model_dir)
        run.assert_called_once()
        self.assertEqual(setup.missing_model_files(self.model_dir), [])

    def test_complete_model_reused_without_downloader(self):
        self.write_model()
        with patch.object(setup, 'modelscope_executable') as executable, patch.object(setup.subprocess, 'run') as run:
            setup.ensure_qwen_model(model_dir=self.model_dir)
        executable.assert_not_called()
        run.assert_not_called()

    def test_partial_model_triggers_download(self):
        self.model_dir.mkdir(parents=True)
        (self.model_dir/'README.md').write_text('partial download')
        with patch.object(setup, 'modelscope_executable', return_value='modelscope'), patch.object(setup.subprocess, 'run') as run:
            run.side_effect = lambda *a, **k: (self.write_model() or type('Result', (), {'returncode':0})())
            setup.ensure_qwen_model(model_dir=self.model_dir)
        run.assert_called_once()

    def test_missing_shard_and_empty_weight_detected(self):
        self.write_model(sharded=True)
        self.assertEqual(setup.missing_model_files(self.model_dir), [])
        (self.model_dir/'model-00002.safetensors').write_bytes(b'')
        self.assertEqual(setup.missing_model_files(self.model_dir), ['model-00002.safetensors'])

    def test_download_failure_and_incomplete_download_stop_training(self):
        with patch.object(setup, 'modelscope_executable', return_value='modelscope'), patch.object(setup.subprocess, 'run') as run:
            run.return_value.returncode = 1
            with self.assertRaisesRegex(SystemExit, '下载失败'):
                setup.ensure_qwen_model(model_dir=self.model_dir)
            run.return_value.returncode = 0
            with self.assertRaisesRegex(SystemExit, '仍不完整'):
                setup.ensure_qwen_model(model_dir=self.model_dir)

    def test_dry_run_never_creates_directory_or_downloads(self):
        with patch.object(setup, 'modelscope_executable', return_value=None), patch.object(setup.subprocess, 'run') as run:
            setup.ensure_qwen_model(dry_run=True, model_dir=self.model_dir)
        self.assertFalse(self.model_dir.exists())
        run.assert_not_called()

    def test_missing_cli_explained_before_model_loading(self):
        with patch.object(setup, 'modelscope_executable', return_value=None):
            with self.assertRaisesRegex(SystemExit, 'pip install modelscope'):
                setup.ensure_qwen_model(model_dir=self.model_dir)
        self.assertFalse(self.model_dir.exists())

    def test_pipeline_uses_same_preflight(self):
        with patch.object(pipeline, 'prepare_qwen_model') as prepare:
            pipeline.ensure_qwen_model(True)
        prepare.assert_called_once_with(dry_run=True, model_dir=pipeline.MODEL_DIR)

    def test_standalone_preflight_precedes_data_and_weight_loading(self):
        for name in ('train_qwen35_handwriting_stage1_repro.py', 'train_qwen35_handwriting_stage1_old.py'):
            with self.subTest(script=name):
                tree = ast.parse((ROOT/name).read_text(encoding='utf-8-sig'))
                calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
                preflight = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id=='ensure_qwen_model')
                for node in calls:
                    if isinstance(node.func, ast.Attribute) and node.func.attr in ('from_pretrained', 'load'):
                        self.assertGreater(node.lineno, preflight.lineno)
                        if node.func.attr=='from_pretrained':
                            self.assertTrue(any(k.arg=='local_files_only' and isinstance(k.value, ast.Constant) and k.value.value is True for k in node.keywords))


if __name__ == '__main__':
    unittest.main()
