import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_qwen35_pipeline as pipeline
import training_runs as runs


class TrainingRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.stage = Path(self.temp.name)/'stage1'

    def checkpoint(self, output):
        output.mkdir(parents=True, exist_ok=True)
        for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'training_info.json'):
            (output/name).write_text('{}', encoding='utf-8')
        (output/'model.safetensors').write_bytes(b'checkpoint')

    def completed(self):
        output = runs.create_training_run(self.stage)
        self.checkpoint(output)
        runs.complete_training_run(self.stage, output)
        return output

    def test_new_runs_have_unique_directories_and_preserve_old_weights(self):
        first = self.completed()
        original = (first/'model.safetensors').read_bytes()
        second = runs.create_training_run(self.stage)
        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, second.parent)
        self.assertEqual((first/'model.safetensors').read_bytes(), original)
        self.assertEqual(runs.latest_training_run(self.stage), first)

    def test_explicit_existing_output_is_rejected_without_modification(self):
        output = self.completed()
        with self.assertRaises(FileExistsError):
            runs.create_training_run(self.stage, output)
        self.assertEqual((output/'model.safetensors').read_bytes(), b'checkpoint')
        empty = self.stage/'empty'
        empty.mkdir()
        with self.assertRaises(FileExistsError):
            runs.create_training_run(self.stage, empty)

    def test_pointer_updates_only_after_completed_checkpoint(self):
        first = self.completed()
        second = runs.create_training_run(self.stage)
        (second/'training_info.json').write_text('{}')
        with self.assertRaises(FileNotFoundError):
            runs.complete_training_run(self.stage, second)
        self.assertFalse((second/'TRAIN_DONE').exists())
        self.assertEqual(runs.latest_training_run(self.stage), first)
        self.checkpoint(second)
        runs.complete_training_run(self.stage, second)
        self.assertEqual(runs.latest_training_run(self.stage), second)
        self.assertTrue((first/'model.safetensors').exists())

    def test_pointer_write_failure_preserves_previous_selection(self):
        first = self.completed()
        second = runs.create_training_run(self.stage)
        self.checkpoint(second)
        with patch.object(runs.os, 'replace', side_effect=OSError('simulated interruption')):
            with self.assertRaises(OSError):
                runs.complete_training_run(self.stage, second)
        self.assertEqual(runs.latest_training_run(self.stage), first)
        self.assertEqual(list(self.stage.glob('.latest-*.tmp')), [])

    def test_incomplete_first_run_cannot_be_used_by_stage_two(self):
        output = runs.create_training_run(self.stage)
        self.checkpoint(output)
        with self.assertRaises(FileNotFoundError):
            runs.latest_training_run(self.stage)

    def test_legacy_checkpoint_still_loads_without_being_moved(self):
        self.checkpoint(self.stage)
        original = set(self.stage.iterdir())
        self.assertEqual(runs.latest_training_run(self.stage), self.stage.resolve())
        runs.create_training_run(self.stage)
        self.assertTrue(original.issubset(set(self.stage.iterdir())))
        self.assertEqual(runs.latest_training_run(self.stage), self.stage.resolve())

    def test_default_pointer_is_relative_and_evaluation_resolves_alias(self):
        output = self.completed()
        record = json.loads((self.stage/'latest.json').read_text(encoding='utf-8'))
        self.assertFalse(Path(record['model_dir']).is_absolute())
        self.assertEqual(runs.resolve_model_path(self.stage), output)
        self.assertEqual(runs.resolve_model_path(output), output)
        ordinary = Path('models/qwen3.5-0.8b')
        self.assertEqual(runs.resolve_model_path(ordinary), ordinary)

    def test_broken_published_checkpoint_is_not_silently_skipped(self):
        output = self.completed()
        (output/'model.safetensors').unlink()
        with patch.object(pipeline, 'QWEN_STAGE1_DIR', self.stage):
            with self.assertRaises(FileNotFoundError):
                pipeline.get_stage(1)

    def test_pipeline_ignores_partial_runs_then_finds_child_output(self):
        partial = runs.create_training_run(self.stage)
        self.checkpoint(partial)
        def child(*args, **kwargs):
            self.last = self.completed()
            return SimpleNamespace(returncode=0)
        state = {}
        with patch.object(pipeline, 'QWEN_STAGE1_DIR', self.stage), patch.object(pipeline.subprocess, 'run', side_effect=child) as process, patch.object(pipeline, 'save_state'), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(pipeline.outputs_exist(pipeline.get_stage(1)))
            pipeline.run_stage(1, state, False)
            self.assertEqual(pipeline.get_stage(1)['model'], self.last)
            pipeline.validate_previous_stages(2)
            pipeline.run_stage(1, state, False)
        process.assert_called_once()
        self.assertEqual(state['1']['status'], 'skipped')


if __name__ == '__main__':
    unittest.main()
