import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from model_training.datasetMethods import AudioDatasetFromCSV, build_class_balanced_sampler

SR = 16000
MAX_LEN = 1600


def _write(path, n_samples):
    audio = np.linspace(-0.5, 0.5, n_samples, dtype=np.float32)  # ramp → crop position is visible
    sf.write(path, audio, SR)
    return audio


@pytest.fixture
def split_csv(tmp_path):
    short = _write(tmp_path / "short.wav", 1000)
    long_ = _write(tmp_path / "long.wav", 4000)
    rows = [
        {"file_path": str(tmp_path / "short.wav"), "label": "bonafide", "split": "train"},
        {"file_path": str(tmp_path / "long.wav"), "label": "deepfake", "split": "train"},
        {"file_path": str(tmp_path / "long.wav"), "label": "deepfake", "split": "val"},
    ]
    csv = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv, short, long_


def test_short_clip_is_right_padded_and_masked(split_csv):
    csv, short, _ = split_csv
    item = AudioDatasetFromCSV(csv, split="train", max_length=MAX_LEN)[0]

    assert item["input_values"].shape == (MAX_LEN,)
    assert item["attention_mask"][:1000].all()
    assert not item["attention_mask"][1000:].any()
    assert torch.all(item["input_values"][1000:] == 0)
    np.testing.assert_allclose(item["input_values"][:1000].numpy(), short, atol=1e-4)
    assert item["labels"].tolist() == [1]  # bonafide = 1


def test_long_clip_is_cropped_to_window(split_csv):
    csv, _, _ = split_csv
    item = AudioDatasetFromCSV(csv, split="train", max_length=MAX_LEN)[1]

    assert item["input_values"].shape == (MAX_LEN,)
    assert item["attention_mask"].all()
    assert item["labels"].tolist() == [0]  # deepfake = 0


def test_validation_uses_deterministic_center_crop(split_csv):
    csv, _, long_ = split_csv
    ds = AudioDatasetFromCSV(csv, split="val", max_length=MAX_LEN)

    first, second = ds[0]["input_values"], ds[0]["input_values"]

    start = (len(long_) - MAX_LEN) // 2
    torch.testing.assert_close(first, second)
    np.testing.assert_allclose(first.numpy(), long_[start:start + MAX_LEN], atol=1e-4)


def test_split_filter(split_csv):
    csv, _, _ = split_csv
    assert len(AudioDatasetFromCSV(csv, split="train")) == 2
    assert len(AudioDatasetFromCSV(csv, split="val")) == 1
    assert len(AudioDatasetFromCSV(csv, split="test")) == 0


class _Labels:
    def __init__(self, labels):
        self.labels_array = np.array(labels)


def test_sampler_balances_imbalanced_classes():
    labels = [1] * 900 + [0] * 100  # 90% bonafide
    sampler = build_class_balanced_sampler(_Labels(labels))
    torch.manual_seed(0)

    drawn = np.array(labels)[list(sampler)]

    assert len(drawn) == len(labels)
    assert 0.45 < drawn.mean() < 0.55


def test_sampler_rejects_missing_class():
    with pytest.raises(ValueError):
        build_class_balanced_sampler(_Labels([1, 1, 1]))
