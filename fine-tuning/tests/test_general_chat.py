import json
from pathlib import Path
import tempfile
import unittest

from data_tools.general_chat import parse_conversations, export, load_data, encode_turn


class FakeTokenizer:
    eos_token_id = 9

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return {'input_ids': [1] * sum(len(m['content']) for m in messages)}


class GeneralChatTests(unittest.TestCase):
    def test_history_and_blank_conversation_boundaries(self):
        rows = parse_conversations('问：你好\n答：你好♥\n问：刚才说什么\n答：你好\n\n问：2+2\n答：4\n第二行')
        self.assertEqual([len(r) for r in rows], [4, 2])
        self.assertEqual(rows[0][1]['content'], '你好♥')
        self.assertEqual(rows[1][-1]['content'], '4\n第二行')

    def test_reject_incomplete_or_invalid_roles(self):
        for text in ['答：你好', '问：你好', '问：一\n问：二\n答：三']:
            with self.assertRaises(ValueError):
                parse_conversations(text)

    def test_dedup_split_reproducible_and_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source.txt', Path(directory) / 'out.jsonl'
            source.write_text('问：一\n答：♥\n\n问：二\n答：好\n\n问：一\n答：♥', encoding='utf-8')
            first = export(source, output)
            self.assertEqual(len(first), 2)
            self.assertEqual({r['split'] for r in first}, {'train', 'validation'})
            self.assertEqual(export(source, output), first)
            self.assertEqual(load_data(output), first)

    def test_encode_last_answer_only_with_real_history(self):
        tok = FakeTokenizer()
        messages = [{'role': role, 'content': content} for role, content in
                    [('user', 'A'), ('assistant', 'B'), ('user', 'C'), ('assistant', 'DEF')]]
        tokens, prefix = encode_turn(tok, 'S', messages, 1, 30)
        self.assertEqual(tok.messages[1:], messages[:3])
        self.assertEqual(tokens[prefix:], [0, 1, 2, tok.eos_token_id])

    def test_overlength_drops_old_pairs_then_skips_whole_target(self):
        tok = FakeTokenizer()
        messages = [{'role': role, 'content': content} for role, content in
                    [('user', 'AAAA'), ('assistant', 'BBBB'), ('user', 'C'), ('assistant', 'D')]]
        tokens, prefix = encode_turn(tok, 'S', messages, 1, 4)
        self.assertEqual(tok.messages, [{'role': 'system', 'content': 'S'}, messages[2]])
        self.assertEqual(len(tokens), 4)
        self.assertIsNone(encode_turn(tok, 'S', messages, 1, 3))


if __name__ == '__main__':
    unittest.main()
