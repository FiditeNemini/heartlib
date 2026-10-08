"""Small loader regression tests; no checkpoint downloads or GPUs required."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from safetensors.torch import save_file
from huggingface_hub.utils import LocalEntryNotFoundError
from heartlib.heartcodec.modeling_heartcodec import HeartCodec


class FakeEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))


class SplitLoadingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'encoder_config.json').write_text(json.dumps({'version': 'vqv12'}))
        self.model = nn.Module()
        self.model.decoder = nn.Linear(2, 2)
        self.model.encoder = None
        self.model.config = type('Config', (), {})()

    def tearDown(self):
        self.temp.cleanup()

    def load(self, state, info=None):
        save_file(state, str(self.root / 'encoder.safetensors'))
        with patch.object(HeartCodec, 'from_pretrained', return_value=(self.model, info or {})), \
             patch('heartlib.heartcodec.modeling_heartcodec.HeartCodecEncoder', FakeEncoder):
            return HeartCodec.from_encoder_decoder_pretrained('decoder', self.root)

    def test_strict_load_and_eval(self):
        model = self.load({'encoder.weight': torch.ones(2)})
        self.assertTrue(torch.equal(model.encoder.weight, torch.ones(2)))
        self.assertFalse(model.training)
        self.assertFalse(model.encoder.training)
        self.assertEqual(model.config.encoder_config['version'], 'vqv12')

    def test_config_json_preferred(self):
        (self.root / 'config.json').write_text(json.dumps({'version': 'from-config-json'}))
        model = self.load({'encoder.weight': torch.ones(2)})
        self.assertEqual(model.config.encoder_config['version'], 'from-config-json')

    def test_hub_falls_back_to_encoder_config(self):
        save_file({'encoder.weight': torch.ones(2)}, str(self.root / 'encoder.safetensors'))

        def download(repo_id, filename, **kwargs):
            if filename == 'config.json':
                raise LocalEntryNotFoundError('missing')
            return str(self.root / filename)

        with patch.object(HeartCodec, 'from_pretrained', return_value=(self.model, {})), \
             patch('heartlib.heartcodec.modeling_heartcodec.HeartCodecEncoder', FakeEncoder), \
             patch('heartlib.heartcodec.modeling_heartcodec.hf_hub_download', side_effect=download) as hub:
            model = HeartCodec.from_encoder_decoder_pretrained('decoder', 'org/encoder')
        self.assertEqual(model.config.encoder_config['version'], 'vqv12')
        self.assertEqual([c.args[1] for c in hub.call_args_list],
                         ['config.json', 'encoder_config.json', 'encoder.safetensors'])

    def test_decoder_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            self.load({'encoder.weight': torch.ones(2)}, {'missing_keys': ['decoder.weight']})

    def test_decoder_tensor_in_encoder_rejected(self):
        with self.assertRaises(ValueError):
            self.load({'decoder.weight': torch.ones(2)})

    def test_wrong_encoder_shape_rejected(self):
        with self.assertRaises(RuntimeError):
            self.load({'encoder.weight': torch.ones(3)})


if __name__ == '__main__':
    unittest.main()
