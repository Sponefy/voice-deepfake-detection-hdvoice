"""Shared fixtures. Tests run on CPU and never download the real 300M model:
Wav2Vec2Model.from_pretrained is replaced with a tiny, randomly initialised Wav2Vec2."""

import pytest
import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

from model_training import detector as detector_module

TINY_CONFIG = Wav2Vec2Config(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=64,
    conv_dim=(16, 16),
    conv_stride=(5, 2),
    conv_kernel=(10, 3),
    num_conv_pos_embeddings=16,
    num_conv_pos_embedding_groups=2,
    feat_extract_norm="layer",      # as in XLS-R
    do_stable_layer_norm=True,
)


class _TinyWav2Vec2:
    @staticmethod
    def from_pretrained(_name):
        torch.manual_seed(0)
        return Wav2Vec2Model(TINY_CONFIG)


@pytest.fixture
def tiny_wav2vec2(monkeypatch):
    monkeypatch.setattr(detector_module, "Wav2Vec2Model", _TinyWav2Vec2)
    return TINY_CONFIG


@pytest.fixture
def make_detector(tiny_wav2vec2):
    def _make(**kwargs):
        return detector_module.Wav2Vec2DeepfakeDetector(model_name="tiny", **kwargs)
    return _make
