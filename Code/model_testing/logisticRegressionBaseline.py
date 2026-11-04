"""
D2 z analiza_detektor_deepfake'ów.md — banalna regresja logistyczna jako
baseline. Jeśli trywialne cechy (długość, RMS, ZCR, spectral centroid…) dają
wysokie AUC, to dataset jest separowalny po cechach NISKOPOZIOMOWYCH —
czyli model Wav2Vec2 nie musiał uczyć się żadnej "deepfake-ness", tylko domeny.

Dwa eksperymenty:
  E1 — LR na samej długości pliku (cała baza, szybkie).
  E2 — LR na pakiecie cech low-level (subset 10k plików, wolniejsze).
Plus ewaluacja tego samego LR na OOD (recordings/real + recordings/fake).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from projectMethods import find_project_root, get_dataset_split_csv  # noqa: E402

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

CSV_PATH = str(get_dataset_split_csv(3))
RECORDINGS_ROOT = find_project_root() / "AudioFiles" / "recordings"

# Subsampling dla E2 (drogie cechy spektralne)
E2_SAMPLES_PER_SPLIT = {"train": 5000, "val": 2500, "test": 2500}
RANDOM_STATE = 42


# ============================================================================
# Cechy
# ============================================================================

def duration_from_header(path: str) -> float:
    """Szybkie odczytanie długości z nagłówka (bez dekodowania audio)."""
    try:
        return float(sf.info(path).duration)
    except Exception:
        return np.nan


def low_level_features(path: str) -> dict:
    """
    Wczytuje audio i liczy kilka cech low-level. Zwraca dict (NaN-y przy błędzie).

    Cechy:
      - duration:        sekundy
      - rms:             średnia energia
      - zcr:             zero-crossing rate
      - spec_centroid:   średnia spektralnego centroidu
      - spec_rolloff:    średni rolloff 85%
      - spec_flatness:   średnia spektralnej płaskości
      - dc_offset:       absolutna średnia próbki (DC)
    """
    try:
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        n = len(audio)
        duration = n / sr

        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
        # ZCR
        sign_changes = np.sum(np.abs(np.diff(np.sign(audio)))) / 2.0
        zcr = float(sign_changes / max(n - 1, 1))
        # DC
        dc = float(np.abs(audio.mean()))

        # Spectral — przez prostą FFT, jeden segment (uśredniony po ramkach)
        import librosa
        # Dla bardzo krótkich plików zmniejsz n_fft, żeby uniknąć paddingu librosy
        n_fft = 1024 if n >= 1024 else 256
        hop = n_fft // 4
        if n < n_fft:
            audio_padded = np.pad(audio, (0, n_fft - n))
        else:
            audio_padded = audio
        sc = librosa.feature.spectral_centroid(y=audio_padded, sr=sr, n_fft=n_fft, hop_length=hop)
        sr_rolloff = librosa.feature.spectral_rolloff(y=audio_padded, sr=sr, n_fft=n_fft, hop_length=hop)
        sf_flatness = librosa.feature.spectral_flatness(y=audio_padded, n_fft=n_fft, hop_length=hop)

        return {
            "duration": duration,
            "rms": rms,
            "zcr": zcr,
            "spec_centroid": float(sc.mean()),
            "spec_rolloff": float(sr_rolloff.mean()),
            "spec_flatness": float(sf_flatness.mean()),
            "dc_offset": dc,
        }
    except Exception as e:
        return {
            "duration": np.nan, "rms": np.nan, "zcr": np.nan,
            "spec_centroid": np.nan, "spec_rolloff": np.nan,
            "spec_flatness": np.nan, "dc_offset": np.nan,
        }


# ============================================================================
# E1 — LR na samej długości, cała baza
# ============================================================================

def experiment_1_length_only(df: pd.DataFrame):
    print()
    print("=" * 70)
    print("E1 — LR na samej długości pliku (CAŁA baza)")
    print("=" * 70)
    print("Czytam długości plików z nagłówków…")
    durations = []
    for p in tqdm(df["file_path"].values):
        durations.append(duration_from_header(p))
    df = df.copy()
    df["duration"] = durations
    df = df.dropna(subset=["duration"])

    df["y"] = (df["label"] == "bonafide").astype(np.int64)
    X_train = df[df["split"] == "train"][["duration"]].values
    y_train = df[df["split"] == "train"]["y"].values
    X_val   = df[df["split"] == "val"][["duration"]].values
    y_val   = df[df["split"] == "val"]["y"].values
    X_test  = df[df["split"] == "test"][["duration"]].values
    y_test  = df[df["split"] == "test"]["y"].values

    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(scaler.transform(X_train), y_train)

    for name, X, y in [("train", X_train, y_train),
                       ("val",   X_val,   y_val),
                       ("test",  X_test,  y_test)]:
        Xs = scaler.transform(X)
        prob = clf.predict_proba(Xs)[:, 1]
        pred = (prob >= 0.5).astype(int)
        auc = roc_auc_score(y, prob)
        acc = accuracy_score(y, pred)
        bona_acc = accuracy_score(y[y == 1], pred[y == 1])
        spoof_acc = accuracy_score(y[y == 0], pred[y == 0])
        print(f"  {name:5s}  n={len(y):>6}  AUC={auc:.4f}  acc={acc:.4f}  "
              f"bona={bona_acc:.4f}  spoof={spoof_acc:.4f}")

    print(f"  współczynnik LR (duration): {clf.coef_[0][0]:+.4f}  intercept={clf.intercept_[0]:+.4f}")
    return scaler, clf


# ============================================================================
# E2 — LR na pakiecie cech, na subsetcie
# ============================================================================

FEATURE_NAMES = ["duration", "rms", "zcr", "spec_centroid",
                 "spec_rolloff", "spec_flatness", "dc_offset"]


def sample_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Subsampling z zachowaniem stratyfikacji label×split."""
    rng = np.random.RandomState(RANDOM_STATE)
    parts = []
    for split, n_target in E2_SAMPLES_PER_SPLIT.items():
        sub = df[df["split"] == split]
        # stratify per label
        for lbl in ("bonafide", "deepfake"):
            slab = sub[sub["label"] == lbl]
            n_take = min(len(slab), n_target // 2)
            if n_take > 0:
                idx = rng.choice(len(slab), size=n_take, replace=False)
                parts.append(slab.iloc[idx])
    return pd.concat(parts).reset_index(drop=True)


def compute_features_for_df(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Liczę cechy low-level dla {len(df)} plików…")
    rows = []
    for p in tqdm(df["file_path"].values):
        rows.append(low_level_features(p))
    feats = pd.DataFrame(rows)
    out = pd.concat([df.reset_index(drop=True), feats], axis=1)
    return out.dropna(subset=FEATURE_NAMES)


def fit_lr_on_features(df_feat: pd.DataFrame, feature_cols: list, label: str):
    print(f"\n--- LR na cechach: {feature_cols} ---")
    df_feat = df_feat.copy()
    df_feat["y"] = (df_feat["label"] == "bonafide").astype(np.int64)

    X_train = df_feat[df_feat["split"] == "train"][feature_cols].values
    y_train = df_feat[df_feat["split"] == "train"]["y"].values
    X_val   = df_feat[df_feat["split"] == "val"][feature_cols].values
    y_val   = df_feat[df_feat["split"] == "val"]["y"].values
    X_test  = df_feat[df_feat["split"] == "test"][feature_cols].values
    y_test  = df_feat[df_feat["split"] == "test"]["y"].values

    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(scaler.transform(X_train), y_train)

    for name, X, y in [("train", X_train, y_train),
                       ("val",   X_val,   y_val),
                       ("test",  X_test,  y_test)]:
        Xs = scaler.transform(X)
        prob = clf.predict_proba(Xs)[:, 1]
        pred = (prob >= 0.5).astype(int)
        auc = roc_auc_score(y, prob)
        acc = accuracy_score(y, pred)
        bona_acc = accuracy_score(y[y == 1], pred[y == 1])
        spoof_acc = accuracy_score(y[y == 0], pred[y == 0])
        print(f"  {name:5s}  n={len(y):>6}  AUC={auc:.4f}  acc={acc:.4f}  "
              f"bona={bona_acc:.4f}  spoof={spoof_acc:.4f}")

    print("  współczynniki:")
    for f, w in zip(feature_cols, clf.coef_[0]):
        print(f"    {f:<14} {w:+.4f}")
    print(f"  intercept={clf.intercept_[0]:+.4f}")
    return scaler, clf


# ============================================================================
# OOD eval
# ============================================================================

def evaluate_ood(scaler, clf, feature_cols: list):
    print()
    print("=" * 70)
    print(f"OOD (recordings/) — ten sam LR na zewnętrznych nagraniach")
    print(f"  cechy: {feature_cols}")
    print("=" * 70)

    AUDIO_EXTS = {".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg"}
    items = []
    for root, lbl in [(RECORDINGS_ROOT / "real", "bonafide"),
                      (RECORDINGS_ROOT / "fake", "deepfake")]:
        if not root.is_dir():
            continue
        for f in sorted(root.rglob("*")):
            if f.suffix.lower() in AUDIO_EXTS:
                items.append({"file_path": str(f), "label": lbl,
                              "source": f.parent.name})

    if not items:
        print("(brak plików w recordings/)")
        return
    df = pd.DataFrame(items)
    feats = pd.DataFrame([low_level_features(p) for p in df["file_path"]])
    df = pd.concat([df, feats], axis=1).dropna(subset=feature_cols)
    df["y"] = (df["label"] == "bonafide").astype(np.int64)

    X = scaler.transform(df[feature_cols].values)
    prob = clf.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)

    print(f"\n{'true':<10}{'source':<14}{'plik':<26}{'P(bonafide)':>12}  ✓/✗")
    print("-" * 70)
    for i, row in df.reset_index(drop=True).iterrows():
        correct = pred[i] == row["y"]
        print(f"{row['label']:<10}{row['source']:<14}{Path(row['file_path']).name:<26}"
              f"{prob[i]:>12.4f}  {'✓' if correct else '✗'}")

    y = df["y"].values
    if len(set(y)) > 1:
        print(f"\nOOD AUC={roc_auc_score(y, prob):.4f}  "
              f"acc={accuracy_score(y, pred):.4f}  "
              f"bona={accuracy_score(y[y==1], pred[y==1]):.4f}  "
              f"spoof={accuracy_score(y[y==0], pred[y==0]):.4f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # E1: full data, length-only
    scaler1, clf1 = experiment_1_length_only(df)

    # E2: subset, multi-feature
    print()
    print("=" * 70)
    print("E2 — LR na zestawie cech low-level (SUBSET)")
    print("=" * 70)
    sub_df = sample_subset(df)
    print("Subset (po klasie × split):")
    print(sub_df.groupby(["split", "label"]).size().unstack(fill_value=0))
    feat_df = compute_features_for_df(sub_df)

    # Wariant: wszystkie cechy
    scaler2_all, clf2_all = fit_lr_on_features(feat_df, FEATURE_NAMES, label="all")
    # Wariant: wszystkie cechy POZA długością — czysty test "czy widmo wystarczy"
    no_dur = [f for f in FEATURE_NAMES if f != "duration"]
    scaler2_nodur, clf2_nodur = fit_lr_on_features(feat_df, no_dur, label="no-duration")

    # OOD ewaluacja
    evaluate_ood(scaler1, clf1, ["duration"])
    evaluate_ood(scaler2_all, clf2_all, FEATURE_NAMES)
    evaluate_ood(scaler2_nodur, clf2_nodur, no_dur)


if __name__ == "__main__":
    main()