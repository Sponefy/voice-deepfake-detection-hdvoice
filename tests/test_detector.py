import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from model_training.detector import DeepfakeDetectorTrainer, compute_metrics

SAMPLES = 1600  # 0.1 s at 16 kHz


@pytest.mark.parametrize("pooling_mode", ["mean", "max", "attention"])
def test_forward_returns_two_logits_per_clip(make_detector, pooling_mode):
    model = make_detector(pooling_mode=pooling_mode).eval()
    audio = torch.randn(3, SAMPLES)
    mask = torch.ones(3, SAMPLES, dtype=torch.long)
    mask[1, SAMPLES // 2:] = 0  # right padding

    with torch.no_grad():
        logits = model(audio, attention_mask=mask)

    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()


def test_freeze_encoder_leaves_only_head_trainable(make_detector):
    model = make_detector(freeze_encoder=True)

    assert not any(p.requires_grad for p in model.wav2vec2.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())


@pytest.mark.parametrize("pooling_mode", ["mean", "max", "attention"])
def test_masked_pooling_ignores_padding(make_detector, pooling_mode):
    model = make_detector(pooling_mode=pooling_mode).eval()
    hidden = torch.randn(1, 6, 32)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)

    garbage = hidden.clone()
    garbage[:, 4:] = 1e4  # padded frames with extreme values

    with torch.no_grad():
        clean = model.pool_features(hidden, mask)
        noisy = model.pool_features(garbage, mask)

    torch.testing.assert_close(clean, noisy)
    if pooling_mode == "mean":
        torch.testing.assert_close(clean, hidden[:, :4].mean(dim=1))


def test_feature_mask_matches_real_audio_length(make_detector):
    model = make_detector().eval()
    real = [SAMPLES, SAMPLES // 2]
    mask = torch.zeros(2, SAMPLES, dtype=torch.long)
    for i, n in enumerate(real):
        mask[i, :n] = 1

    with torch.no_grad():
        hidden = model.wav2vec2(torch.randn(2, SAMPLES), attention_mask=mask).last_hidden_state
    feat_mask = model._build_feat_mask(mask, hidden)

    expected = model.wav2vec2._get_feat_extract_output_lengths(torch.tensor(real))
    assert feat_mask.sum(dim=1).tolist() == expected.tolist()
    assert feat_mask[0].all()          # full clip: every frame is real
    assert not feat_mask[1, -1]        # padded clip: last frame is padding


def test_compute_metrics_per_class_accuracy():
    labels = np.array([1, 1, 0, 0])
    # logits: first bonafide wrong, everything else right; scores still rank perfectly
    logits = np.array([[0.6, 0.4], [-2.0, 2.0], [3.0, -3.0], [2.0, -2.0]])

    m = compute_metrics(logits, labels)

    assert m["accuracy"] == pytest.approx(0.75)
    assert m["bonafide_accuracy"] == pytest.approx(0.5)
    assert m["spoof_accuracy"] == pytest.approx(1.0)
    assert m["auc"] == pytest.approx(1.0)


def test_compute_metrics_single_class_does_not_crash():
    m = compute_metrics(np.array([[0.0, 1.0], [0.0, 2.0]]), np.array([1, 1]))
    assert m["auc"] == 0.0
    assert m["spoof_accuracy"] == 0.0


def _loader(n=8):
    torch.manual_seed(0)
    audio = torch.randn(n, SAMPLES)
    mask = torch.ones(n, SAMPLES, dtype=torch.long)
    labels = torch.tensor([[i % 2] for i in range(n)])

    class Dict(TensorDataset):
        def __getitem__(self, i):
            a, m, l = super().__getitem__(i)
            return {"input_values": a, "attention_mask": m, "labels": l}

    return DataLoader(Dict(audio, mask, labels), batch_size=4)


def test_trainer_runs_and_saves_best_model(make_detector, tmp_path):
    model = make_detector(freeze_encoder=True)
    trainer = DeepfakeDetectorTrainer(
        model, _loader(), _loader(), torch.device("cpu"), gradient_accumulation_steps=2,
    )
    save_path = tmp_path / "best.pt"

    trainer.train(num_epochs=1, save_path=str(save_path))

    metrics = trainer.validate()
    assert np.isfinite(metrics["loss"])
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert save_path.exists()
    state = torch.load(save_path, map_location="cpu")
    assert set(state) == set(model.state_dict())


def test_early_stopping_stops_when_selection_metric_stalls(make_detector, tmp_path, capsys):
    model = make_detector(freeze_encoder=True)
    trainer = DeepfakeDetectorTrainer(model, _loader(), _loader(), torch.device("cpu"))
    trainer.validate = lambda loader=None, desc="": {   # constant AUC -> never improves
        "loss": 0.5, "accuracy": 0.5, "auc": 0.5,
        "bonafide_accuracy": 0.5, "spoof_accuracy": 0.5,
    }

    trainer.train(num_epochs=10, save_path=str(tmp_path / "m.pt"), patience=2)

    out = capsys.readouterr().out
    assert "Early stopping" in out
    assert "Epoch 3/10" in out and "Epoch 4/10" not in out
