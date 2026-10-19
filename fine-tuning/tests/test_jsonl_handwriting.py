import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data_tools import qwen
from project_paths import ROOT, QWEN_HANDWRITING, QWEN_DRAWING


def example(index,split):
    codes=list(range(16))
    return {'id':f'row{index}','sample_id':f'sample{index}','character':qwen.CHARS[index],'label':index,
            'split':split,'split_order':0,'variant':0,'codes':codes,'system':'请生成笔迹编号。',
            'user':f'请写一个{qwen.CHARS[index]}。','assistant':'<DRAW>'+''.join(f'<M{i:03d}>' for i in codes)+'<END_DRAW>'}


class JsonlHandwritingTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)/'data.jsonl'
        self.rows=[example(0,'train'),example(1,'validation')]

    def write(self,rows):
        self.path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
        return qwen.read_handwriting(self.path)

    def test_corpus_has_consistent_codes_and_splits_across_stages(self):
        stage1=qwen.read_handwriting(QWEN_HANDWRITING)
        draw=qwen.read_handwriting(QWEN_DRAWING)
        self.assertEqual((len(stage1.labels),len(stage1.train),len(stage1.validation)),(799,686,113))
        self.assertEqual(stage1.labels,draw.labels)
        self.assertEqual(stage1.codes,draw.codes)
        self.assertEqual(stage1.train,draw.train)
        self.assertEqual(stage1.validation,draw.validation)
        self.assertEqual(len(stage1.examples[0]),3)
        self.assertEqual(len(draw.examples[0]),10)
        self.assertFalse(set(stage1.train)&set(stage1.validation))
        self.assertEqual(set(stage1.labels[i] for i in stage1.train),set(range(36)))
        self.assertEqual(set(stage1.labels[i] for i in stage1.validation),set(range(36)))
        self.assertTrue(all(any('\u4e00'<=c<='\u9fff' for c in row['user']) for group in stage1.examples+draw.examples for row in group))

    def test_raw_jsonl_loads_coordinates_and_metadata(self):
        from aligned_digit_data import load_raw,load
        raw=load_raw()
        self.assertEqual(raw['points'].shape,(800,64,3))
        self.assertEqual(str(raw['points'].dtype),'float32')
        xy,labels,held=load()
        self.assertEqual(xy.shape,(799,64,2))
        self.assertEqual(len(labels),len(held))
        self.assertFalse(list((ROOT/'data').rglob('*.npz')))

    def test_invalid_codes_labels_and_targets_rejected(self):
        changes=[{'codes':[0]*15},{'codes':[256]*16},{'codes':[True]*16},
                 {'character':'Z'},{'assistant':'<DRAW><M000><END_DRAW>'},{'user':''},{'variant':-1}]
        for change in changes:
            with self.subTest(change=change):
                rows=copy.deepcopy(self.rows);rows[0].update(change)
                with self.assertRaises(ValueError):self.write(rows)

    def test_same_source_cannot_leak_into_both_splits(self):
        leaked=copy.deepcopy(self.rows[0]);leaked.update(id='leaked',variant=1,split='validation')
        with self.assertRaises(ValueError):self.write(self.rows+[leaked])

    def test_duplicate_ids_and_missing_variants_rejected(self):
        with self.assertRaises(ValueError):self.write(self.rows+[self.rows[0]])
        variant=copy.deepcopy(self.rows[0]);variant.update(id='variant',variant=2)
        with self.assertRaises(ValueError):self.write(self.rows+[variant])

    def test_edited_question_and_answer_are_used_by_encoder(self):
        dataset=self.write(self.rows)
        row=dataset.examples[0][0]
        class Tokenizer:
            eos_token_id=99
            def apply_chat_template(self,messages,**kwargs):
                self.messages=messages
                return {'input_ids':[1,2,3]}
            def encode(self,text,**kwargs):
                self.answer=text
                return [10]+list(range(20,36))+[11]
            def convert_tokens_to_ids(self,text):return 10
        tokenizer=Tokenizer()
        row['user']='麻烦你手写数字零。'
        sequence,prefix,start=qwen.encode_example(tokenizer,row)
        self.assertEqual(tokenizer.messages[-1]['content'],row['user'])
        self.assertEqual(tokenizer.answer,row['assistant'])
        self.assertEqual(sequence[start:start+16],list(range(20,36)))
        self.assertEqual((prefix,start,sequence[-1]),(3,4,99))

    def test_changed_codec_requires_explicit_regeneration(self):
        root=Path(self.directory.name)
        source=root/'tracks.jsonl';source.write_text('original')
        codec=root/'codec.pt';codec.write_bytes(b'new codec')
        first=root/'stage1.jsonl';first.write_text('exists')
        mixed=root/'draw.jsonl';mixed.write_text('exists')
        (root/'handwriting_manifest.json').write_text(json.dumps({'source_sha256':qwen.sha256(source),'codec_sha256':'old'}))
        with patch.multiple(qwen,QWEN_HANDWRITING=first,QWEN_DRAWING=mixed,DIGIT_DATA=source,SMOOTH_VQ=codec):
            with self.assertRaisesRegex(ValueError,'export_qwen --force'):
                qwen.ensure_handwriting_data()
        self.assertEqual(first.read_text(),'exists')

    def test_jsonl_fingerprint_survives_windows_line_endings(self):
        self.path.write_bytes(b'{"row":1}\n{"row":2}\n')
        original=qwen.sha256(self.path)
        self.path.write_bytes(b'{"row":1}\r\n{"row":2}\r\n')
        self.assertEqual(qwen.sha256(self.path),original)

    def test_all_qwen_trainers_read_exported_jsonl(self):
        names=['train_qwen35_handwriting_stage1_repro.py','train_qwen35_handwriting_stage1_old.py',
               'train_precision_ablation.py','train_precision_ablation_oldmethod.py',
               'train_chat_letters.py','train_multiturn.py','train_confirmation.py']
        for name in names:
            with self.subTest(script=name):
                tree=ast.parse((ROOT/name).read_text(encoding='utf-8-sig'))
                imports=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
                self.assertIn('data_tools.qwen',imports)
                self.assertNotIn('aligned_digit_data',imports)
                self.assertNotIn('aligned_vq_model',imports)


if __name__=='__main__':
    unittest.main()
