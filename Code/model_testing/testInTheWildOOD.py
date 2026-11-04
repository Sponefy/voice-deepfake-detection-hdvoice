"""
Ewaluacja modelu na zbiorze In-the-Wild (OOD) z V2_mk.

Zbiór: AudioFiles/Datasets/V2_mk/compressed/ood_in_the_wild/
  - bonafide/  (750 plików)
  - deepfake/  (750 plików)

Tworzy tymczasowy CSV w pamięci, wczytuje model wav2vec2_300m_V2_mk.pt,
liczy AUC, accuracy oraz per-klasa accuracy.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from model_training.datasetMethods import AudioDatasetFromCSV
from model_training.detector import Wav2Vec2DeepfakeDetector, DeepfakeDetectorTrainer
from projectMethods import get_device, find_project_root


CONFIG = {
    "model_path": "Models/wav2vec2_300m_V2_mk.pt",
    "model_name": "facebook/wav2vec2-xls-r-300m",
    "pooling_mode": "mean",
    "dropout": 0.3,

    "ood_dir": "AudioFiles/Datasets/V2_mk/compressed/ood_in_the_wild",
    "max_audio_length": 64000,   # 4 s @ 16 kHz — tak jak w treningu
    "batch_size": 16,
    "num_workers": 4,
}


def build_ood_csv(ood_dir: Path) -> Path:
    """Tworzy tymczasowy CSV z plikami z ood_in_the_wild/{bonafide,deepfake}."""
    rows = []
    for cls in ("bonafide", "deepfake"):
        cls_dir = ood_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(cls_dir)
        for f in sorted(cls_dir.rglob("*.wav")):
            rows.append({"file_path": str(f), "label": cls, "split": "test"})

    if not rows:
        raise RuntimeError(f"Brak plików .wav w {ood_dir}")

    df = pd.DataFrame(rows)
    tmp = Path(tempfile.gettempdir()) / "ood_in_the_wild_split.csv"
    df.to_csv(tmp, index=False)
    print(f"CSV ad-hoc: {tmp}")
    print(df.groupby("label").size().to_string())
    print()
    return tmp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=CONFIG["model_path"],
                        help="Ścieżka (lub względna do root projektu) do .pt")
    args = parser.parse_args()

    root = find_project_root()
    device = get_device()
    print()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Brak modelu: {model_path}")
    print(f"Model: {model_path}")

    ood_csv = build_ood_csv(root / CONFIG["ood_dir"])

    test_dataset = AudioDatasetFromCSV(
        csv_file=str(ood_csv),
        split="test",
        max_length=CONFIG["max_audio_length"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    print(f"Test (OOD in-the-wild): {len(test_dataset)} sampli")

    print("\nInicjalizacja modelu...")
    model = Wav2Vec2DeepfakeDetector(
        model_name=CONFIG["model_name"],
        freeze_encoder=False,        # nie ma znaczenia przy inference
        pooling_mode=CONFIG["pooling_mode"],
        dropout=CONFIG["dropout"],
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    # Reużywamy DeepfakeDetectorTrainer.validate (oblicza AUC, acc, per-klasa acc).
    trainer = DeepfakeDetectorTrainer(
        model=model,
        train_loader=test_loader,   # niewykorzystywane w validate()
        val_loader=test_loader,
        device=device,
    )

    metrics = trainer.validate(test_loader, desc="OOD in-the-wild")

    print(f"\n{'=' * 60}")
    print("  OOD IN-THE-WILD — WYNIKI")
    print(f"{'=' * 60}")
    print(f"  Loss:             {metrics['loss']:.4f}")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  AUC:              {metrics['auc']:.4f}")
    print(f"  Bonafide Acc:     {metrics['bonafide_accuracy']:.4f}")
    print(f"  Spoof Acc:        {metrics['spoof_accuracy']:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()