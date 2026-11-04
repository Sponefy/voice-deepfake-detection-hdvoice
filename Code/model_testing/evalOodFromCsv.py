"""
Ewaluacja OOD (In-the-Wild) z KALIBRACJĄ PROGU dla datasetu wersji 5
(dataset_split_balanced_ood.csv).

- próg kalibrowany na splicie `val_ood` (max balanced accuracy = Youden J),
- finalne metryki na rozłącznym splicie `test` (po mówcy, bez leaku),
- raport przy progu 0.5 (argmax) vs progu skalibrowanego + AUC i EER (threshold-free).

AUC jest niezależne od progu — to główna miara. Kalibracja naprawia rozjazd
per-klasa (np. Bonafide 0.99 / Spoof 0.20 przy 0.5), wybierając uczciwy punkt
pracy na danych, których model nie widział w treningu.

Ten sam preprocessing co trening (HD Voice, okno 4 s, attention_mask).
NIE odpalaj równolegle z treningiem na tym samym GPU.

Uruchomienie:
    python Code/model_testing/evalOodFromCsv.py
    python Code/model_testing/evalOodFromCsv.py --model Models/wav2vec2_300m_V2_balanced_ood.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from model_training.datasetMethods import AudioDatasetFromCSV
from model_training.detector import Wav2Vec2DeepfakeDetector
from projectMethods import get_device, find_project_root, get_dataset_split_csv

CONFIG = {
    "model_path": "Models/wav2vec2_300m_not_frozen_V2_balanced_ood_v2_2.pt",
    "model_name": "facebook/wav2vec2-xls-r-300m",
    "pooling_mode": "mean",
    "dropout": 0.3,
    "dataset_version": 5,
    "max_audio_length": 64000,
    "batch_size": 16,
    "num_workers": 4,
}


@torch.no_grad()
def predict(model, loader, device):
    """Zwraca (logits[N,2], labels). Logity (nie softmax) — potrzebne do temperature scalingu."""
    model.eval()
    logits_all, labels = [], []
    use_amp = device.type in ("cuda", "mps")
    from tqdm import tqdm
    for batch in tqdm(loader, desc="predykcja", file=sys.stdout):
        iv = batch["input_values"].to(device)
        am = batch.get("attention_mask")
        if am is not None:
            am = am.to(device)
        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(iv, attention_mask=am)
        logits_all.append(logits.float().cpu().numpy())
        labels.append(batch["labels"].squeeze(1).numpy())
    return np.concatenate(logits_all), np.concatenate(labels)


def scores_from_logits(logits, T=1.0):
    """P(bonafide) z logitów, z opcjonalną temperaturą T (dzieli logity przez T)."""
    z = torch.tensor(np.asarray(logits), dtype=torch.float32) / T
    return torch.softmax(z, dim=1)[:, 1].numpy()


def fit_temperature(logits, labels):
    """Dopasowuje skalar T minimalizujący NLL (cross-entropy) na (logits, labels).
    Grid + zagęszczenie. T>1 oznacza, że model był przepewniony."""
    lg = torch.tensor(np.asarray(logits), dtype=torch.float32)
    lb = torch.tensor(np.asarray(labels), dtype=torch.long)

    def nll(T):
        return F.cross_entropy(lg / float(T), lb).item()

    grid = np.linspace(0.5, 10.0, 96)
    best = min(grid, key=nll)
    fine = np.linspace(max(0.05, best - 0.15), best + 0.15, 61)
    return float(min(fine, key=nll))


def ece(scores, labels, n_bins=15):
    """Expected Calibration Error (binarne). scores = P(bonafide)."""
    pred = (scores >= 0.5).astype(int)
    conf = np.where(pred == 1, scores, 1.0 - scores)   # pewność przewidzianej klasy
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    e, n = 0.0, len(scores)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            e += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return e


def metrics_at(scores, labels, thr):
    """Metryki przy progu: bonafide gdy score >= thr (bonafide=1, spoof=0)."""
    pred = (scores >= thr).astype(int)
    bona = labels == 1
    spoof = labels == 0
    bona_acc = float((pred[bona] == 1).mean()) if bona.any() else 0.0
    spoof_acc = float((pred[spoof] == 0).mean()) if spoof.any() else 0.0
    return {
        "accuracy": float((pred == labels).mean()),
        "balanced_acc": (bona_acc + spoof_acc) / 2,
        "bonafide_acc": bona_acc,
        "spoof_acc": spoof_acc,
    }


def eer(labels, scores):
    """Equal Error Rate (+ próg EER). positive=bonafide=1."""
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return (fpr[i] + fnr[i]) / 2, float(thr[i])


def youden_threshold(labels, scores):
    """Próg maksymalizujący balanced accuracy (J = TPR - FPR)."""
    fpr, tpr, thr = roc_curve(labels, scores)
    i = int(np.argmax(tpr - fpr))
    return float(thr[i])


def make_loader(csv, split, device):
    ds = AudioDatasetFromCSV(csv_file=csv, split=split, max_length=CONFIG["max_audio_length"])
    loader = DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=False,
                        num_workers=CONFIG["num_workers"], pin_memory=(device.type == "cuda"))
    return ds, loader


def report(title, scores, labels, thr):
    m = metrics_at(scores, labels, thr)
    auc = roc_auc_score(labels, scores) if len(set(labels.tolist())) > 1 else float("nan")
    er, _ = eer(labels, scores)
    print(f"\n{title}  (próg={thr:.3f})")
    print(f"  AUC:           {auc:.4f}   (threshold-free)")
    print(f"  EER:           {er:.4f}")
    print(f"  Accuracy:      {m['accuracy']:.4f}")
    print(f"  Balanced Acc:  {m['balanced_acc']:.4f}")
    print(f"  Bonafide Acc:  {m['bonafide_acc']:.4f}")
    print(f"  Spoof Acc:     {m['spoof_acc']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CONFIG["model_path"])
    args = ap.parse_args()

    root = find_project_root()
    device = get_device()
    print()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Brak modelu: {model_path}")
    print(f"Model: {model_path}")

    csv = str(get_dataset_split_csv(CONFIG["dataset_version"]))

    model = Wav2Vec2DeepfakeDetector(
        model_name=CONFIG["model_name"], freeze_encoder=False,
        pooling_mode=CONFIG["pooling_mode"], dropout=CONFIG["dropout"],
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # --- splity ---
    val_ds, val_loader = make_loader(csv, "val_ood", device)
    test_ds, test_loader = make_loader(csv, "test", device)
    print(f"val_ood: {len(val_ds)} sampli (kalibracja)   |   test: {len(test_ds)} sampli (finalny OOD)")

    # predykcje (logity) — val_ood do selekcji progu/temperatury, test do metryk
    test_logits, test_labels = predict(model, test_loader, device)
    if len(val_ds) > 0:
        val_logits, val_labels = predict(model, val_loader, device)
    else:
        val_logits, val_labels = np.empty((0, 2)), np.array([])

    test_scores = scores_from_logits(test_logits)   # T=1 (surowe)

    print(f"\n{'=' * 60}")
    print("  OOD IN-THE-WILD — WYNIKI")
    print(f"{'=' * 60}")

    # --- GŁÓWNA MIARA: AUC/EER na TEST (threshold-free) ---
    # val_ood służy teraz do selekcji modelu (trening) i progu, więc nagłówek
    # liczymy na ROZŁĄCZNYM 'test'. AUC/EER są threshold-free → temperatura ich
    # nie zmienia.
    test_auc = roc_auc_score(test_labels, test_scores)
    test_eer, _ = eer(test_labels, test_scores)
    print(f"\n★ TEST (held-out, {len(test_labels)} sampli — GŁÓWNA MIARA):")
    print(f"    AUC: {test_auc:.4f}    |    EER: {test_eer:.4f}")

    if len(val_labels):
        val_scores = scores_from_logits(val_logits)
        full_scores = np.concatenate([val_scores, test_scores])
        full_labels = np.concatenate([val_labels, test_labels])
        print(f"  (poglądowo — pełny In-the-Wild {len(full_labels)}: "
              f"AUC {roc_auc_score(full_labels, full_scores):.4f} / "
              f"EER {eer(full_labels, full_scores)[0]:.4f}; "
              f"val_ood AUC {roc_auc_score(val_labels, val_scores):.4f} "
              f"— UWAGA: val_ood użyty w selekcji)")

    # --- OPERATING POINT bez kalibracji (pokazuje problem przepewnienia) ---
    report("TEST @ próg 0.5 (argmax, niekalibrowany)", test_scores, test_labels, 0.5)

    if len(val_ds) == 0:
        print("\n⚠ Brak splitu val_ood — uruchom createBalancedOodSplit.py (--ood-val-frac > 0), "
              "by włączyć kalibrację progu.")
        return

    # --- KALIBRACJA progu na surowych score'ach (Youden na val_ood) ---
    thr_raw = youden_threshold(val_labels, val_scores)
    print(f"\nKalibracja (surowa) na val_ood: próg = {thr_raw:.3f}")
    report("TEST @ próg skalibrowany (surowy)", test_scores, test_labels, thr_raw)

    # --- TEMPERATURE SCALING: dopasowanie T na val_ood ---
    # T>1 rozsuwa nasycony softmax → próg ląduje w zdrowym zakresie, a
    # prawdopodobieństwa stają się sensowne (niższe ECE). AUC/EER bez zmian,
    # balanced-acc przy optimum też ten sam — zmienia się skala i kalibracja.
    T = fit_temperature(val_logits, val_labels)
    val_scores_T = scores_from_logits(val_logits, T)
    test_scores_T = scores_from_logits(test_logits, T)
    thr_T = youden_threshold(val_labels, val_scores_T)
    print(f"\nTemperature scaling: T = {T:.3f}  (T>1 ⇒ model był przepewniony)")
    print(f"  ECE test: {ece(test_scores, test_labels):.4f} (surowe) → "
          f"{ece(test_scores_T, test_labels):.4f} (po temperaturze)")
    print(f"  próg po temperaturze = {thr_T:.3f}  (zamiast surowego {thr_raw:.3f})")
    report("TEST @ próg skalibrowany + temperature scaling", test_scores_T, test_labels, thr_T)
    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()