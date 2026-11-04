"""
Kompletny pipeline treningowy detekcji deepfake'ów audio (Wav2Vec2)
Uruchamiać bezpośrednio: python Code/model_training/train.py
"""

import sys
from pathlib import Path

# Dodaj folder Code/ do ścieżki, żeby znaleźć moduły projektu
_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import torch
from torch.utils.data import DataLoader

from model_training.datasetMethods import AudioDatasetFromCSV, build_class_balanced_sampler
from projectMethods import get_device, get_dataset_split_csv, find_project_root
from model_training.detector import Wav2Vec2DeepfakeDetector, DeepfakeDetectorTrainer


# ============================================================================
# KONFIGURACJA
# ============================================================================

CONFIG = {
    # Model
    'model_name': 'facebook/wav2vec2-xls-r-300m',
    'freeze_encoder': False,
    'pooling_mode': 'mean',          # 'mean' | 'max' | 'attention'
    'dropout': 0.3,

    # Dataset
    # wersje dataset:
    # 1 - V2 losowy (stary, z leakiem mic1/mic2)
    # 2 - V2 by_utterance (mic1+mic2 tej samej wypowiedzi razem)
    # 3 - V2 by_speaker (mówcy rozdzieleni między splity)
    # 4 - V2_mk (V2_small + OOD: CodecFake/DFADD/PeoplesSpeech w train)
    # 5 - V2 balans źródeł bonafide PO CZASIE + test = OOD (In-the-Wild)

    'csv_file': str(get_dataset_split_csv(5)),
    'max_audio_length': 64000,       # 4 sekundy przy 16 kHz
                                     # Wcześniej 160000 (10 s) — powodowało
                                     # leak: krótkie bonafide miały dużo
                                     # paddingu, długie deepfake'i 0 zer.
                                     # 4 s ≈ median bonafide (3,75 s).

    # Trening
    'num_epochs': 10, #10
    'batch_size': 16, #8
    'learning_rate': 5e-5,
    'gradient_accumulation_steps': 2,
    'num_workers': 4, #2 
    'label_smoothing': 0.05,         # przeciw overconfidence (przepewnieniu modelu)
    'patience': 3,                   # early stopping po VAL_OOD AUC (None = wyłączone)

    # Zapis
    'save_dir': str(find_project_root() / 'Models'),
    'save_name': 'wav2vec2_300m_not_frozen_V2_balanced_ood_v2_2.pt',

    # Opcje
    'use_wandb': False,
    'run_test_after_training': True,  # ewaluacja na OOD (In-the-Wild), split 'test'
}


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  DEEPFAKE DETECTION — TRENING WAV2VEC2")
    print("=" * 60)

    # --- Device ---
    device = get_device()
    print()

    # --- Datasety ---
    print(f"\nWczytywanie datasetów z: {CONFIG['csv_file']}")
    train_dataset = AudioDatasetFromCSV(
        csv_file=CONFIG['csv_file'],
        split='train',
        transform=None,
        max_length=CONFIG['max_audio_length'],
    )
    val_dataset = AudioDatasetFromCSV(
        csv_file=CONFIG['csv_file'],
        split='val',
        transform=None,
        max_length=CONFIG['max_audio_length'],
    )
    test_dataset = AudioDatasetFromCSV(
        csv_file=CONFIG['csv_file'],
        split='test',
        transform=None,
        max_length=CONFIG['max_audio_length'],
    )
    # val_ood (In-the-Wild, rozłączni mówcy względem test) — SELEKCJA MODELU.
    # In-domain val nasyca się (AUC→1.0) i jest ślepy; OOD val ma zapas i koreluje
    # z generalizacją. Nagłówkowy wynik raportujemy potem na 'test' (rozłączny).
    val_ood_dataset = AudioDatasetFromCSV(
        csv_file=CONFIG['csv_file'],
        split='val_ood',
        transform=None,
        max_length=CONFIG['max_audio_length'],
    )

    print(f"✓ Train:      {len(train_dataset):>6} sampli")
    print(f"✓ Validation: {len(val_dataset):>6} sampli  (in-domain — tylko monitoring luki)")
    print(f"✓ Val OOD:    {len(val_ood_dataset):>6} sampli  (In-the-Wild — SELEKCJA modelu po AUC)")
    print(f"✓ Test (OOD): {len(test_dataset):>6} sampli  (In-the-Wild — held-out, finalny pomiar)")

    # --- DataLoadery ---
    # Train: WeightedRandomSampler równoważy klasy w batchach
    # (CSV ma ~76% bonafide / 24% deepfake — bez tego batch=8 ma w średniej
    # ~6 bonafide + 2 deepfake i model się przeciąża na klasę większościową).
    train_sampler = build_class_balanced_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        sampler=train_sampler,           # zastępuje shuffle=True
        num_workers=CONFIG['num_workers'],
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=(device.type == 'cuda'),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=(device.type == 'cuda'),
    )
    val_ood_loader = DataLoader(
        val_ood_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=(device.type == 'cuda'),
    )

    # --- Model ---
    print("\nInicjalizacja modelu...")
    model = Wav2Vec2DeepfakeDetector(
        model_name=CONFIG['model_name'],
        freeze_encoder=CONFIG['freeze_encoder'],
        pooling_mode=CONFIG['pooling_mode'],
        dropout=CONFIG['dropout'],
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Całkowite parametry:   {total_params:>15,}")
    print(f"✓ Trenowalne parametry:  {trainable_params:>15,}")

    # --- Ścieżka zapisu ---
    save_dir = Path(CONFIG['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(save_dir / CONFIG['save_name'])
    print(f"✓ Najlepszy model będzie zapisany do: {save_path}")

    # --- Wandb (opcjonalnie) ---
    if CONFIG['use_wandb']:
        import wandb
        wandb.init(project="deepfake-detection", config=CONFIG)

    # --- Trainer ---
    trainer = DeepfakeDetectorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=CONFIG['learning_rate'],
        use_wandb=CONFIG['use_wandb'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        label_smoothing=CONFIG['label_smoothing'],
    )

    # --- Trening ---
    # Selekcja best-modelu po VAL_OOD AUC (in-domain val nasyca się i jest ślepy);
    # early stopping po tej samej metryce (pełny fine-tuning bywa szczytuje OOD wcześnie).
    print("\nRozpoczynam trening...")
    trainer.train(
        num_epochs=CONFIG['num_epochs'],
        save_path=save_path,
        select_loader=val_ood_loader,
        patience=CONFIG['patience'],
    )

    # --- Ewaluacja OOD (In-the-Wild) — jedyna uczciwa miara generalizacji ---
    if CONFIG['run_test_after_training']:
        print("\nWczytywanie najlepszego modelu (wg VAL_OOD AUC) do ewaluacji na test...")
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.to(device)

        test_metrics = trainer.validate(test_loader, desc="Test OOD")

        print(f"\n{'=' * 60}")
        print("  WYNIKI OOD (In-the-Wild, held-out)")
        print(f"{'=' * 60}")
        print(f"  Loss:             {test_metrics['loss']:.4f}")
        print(f"  Accuracy:         {test_metrics['accuracy']:.4f}")
        print(f"  AUC:              {test_metrics['auc']:.4f}")
        print(f"  Bonafide Acc:     {test_metrics['bonafide_accuracy']:.4f}")
        print(f"  Spoof Acc:        {test_metrics['spoof_accuracy']:.4f}")
        print(f"{'=' * 60}")

    print("\n✓ Pipeline zakończony.")


if __name__ == '__main__':
    main()
