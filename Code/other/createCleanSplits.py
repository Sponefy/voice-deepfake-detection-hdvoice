"""
Tworzy dwa nowe splity datasetu V1 obok istniejącego (oryginalny zostaje):

1. dataset_split_by_utterance.csv
   Bonafide: split po utterance_id (p225_001) — mic1 i mic2 tej samej wypowiedzi
   zawsze w tym samym splicie. Mówcy mogą się powtarzać między splitami.
   Deepfake: split po pliku (jak dotąd).

2. dataset_split_by_speaker.csv
   Bonafide: split po mówcy (p225) — żaden mówca nie pojawia się w >1 splicie.
   Deepfake: split po pliku w obrębie każdego generatora (zachowuje balans
   generatorów między train/val/test).

Proporcje: 70/15/15, random_state=42.
"""

import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))
from projectMethods import get_dataset_path

AUDIO_EXTS = {".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg"}
UTT_RE = re.compile(r"(p\d+)_(\d+)_mic[12]")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42


def split_items(items, train_r, val_r, test_r, seed):
    """Standardowy 3-way split listy."""
    train, temp = train_test_split(items, test_size=val_r + test_r, random_state=seed)
    val, test = train_test_split(temp, test_size=test_r / (val_r + test_r), random_state=seed)
    return set(train), set(val), set(test)


def collect_audio(folder: Path):
    return [f for f in folder.rglob("*") if f.suffix.lower() in AUDIO_EXTS]


def build_bonafide_by_utterance(bonafide_dir: Path, files):
    """Grupowanie po utterance_id (p225_001), split na grupach."""
    file_to_utt = {}
    utterances = set()
    for f in files:
        m = UTT_RE.match(f.stem)
        if m:
            utt_id = f"{m.group(1)}_{m.group(2)}"
        else:
            # plik bez mic1/mic2 — traktujemy każdy jako osobną grupę
            utt_id = f.stem
        file_to_utt[f] = utt_id
        utterances.add(utt_id)

    train_u, val_u, test_u = split_items(
        sorted(utterances), TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE
    )

    rows = []
    for f, utt in file_to_utt.items():
        if utt in train_u:
            split = "train"
        elif utt in val_u:
            split = "val"
        else:
            split = "test"
        rows.append({"file_path": str(f), "label": "bonafide", "split": split})
    return rows


def build_bonafide_by_speaker(bonafide_dir: Path, files):
    """Split po mówcy (p225) — każdy mówca w dokładnie jednym splicie."""
    file_to_speaker = {}
    speakers = set()
    for f in files:
        m = UTT_RE.match(f.stem)
        speaker = m.group(1) if m else f.parent.name
        file_to_speaker[f] = speaker
        speakers.add(speaker)

    train_s, val_s, test_s = split_items(
        sorted(speakers), TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE
    )

    rows = []
    for f, sp in file_to_speaker.items():
        if sp in train_s:
            split = "train"
        elif sp in val_s:
            split = "val"
        else:
            split = "test"
        rows.append({"file_path": str(f), "label": "bonafide", "split": split})
    return rows, (len(train_s), len(val_s), len(test_s))


def build_deepfake_by_file_per_generator(deepfake_dir: Path):
    """Split po pliku w obrębie każdego generatora — proporcje generatorów zachowane."""
    rows = []
    for gen_dir in sorted(p for p in deepfake_dir.iterdir() if p.is_dir()):
        files = collect_audio(gen_dir)
        train_f, val_f, test_f = split_items(
            files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE
        )
        for f in files:
            if f in train_f:
                split = "train"
            elif f in val_f:
                split = "val"
            else:
                split = "test"
            rows.append({"file_path": str(f), "label": "deepfake", "split": split})
    return rows


def save_and_summarize(rows, out_path: Path, title: str):
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\n=== {title} ===")
    print(f"Zapisano: {out_path}")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))


def main():
    dataset_dir = get_dataset_path()
    out_dir = dataset_dir.parent
    bonafide_dir = dataset_dir / "bonafide"
    deepfake_dir = dataset_dir / "deepfake"

    bonafide_files = collect_audio(bonafide_dir)
    deepfake_rows = build_deepfake_by_file_per_generator(deepfake_dir)

    # 1) by_utterance
    rows = build_bonafide_by_utterance(bonafide_dir, bonafide_files) + deepfake_rows
    save_and_summarize(
        rows,
        out_dir / "dataset_split_by_utterance.csv",
        "BY UTTERANCE (bonafide grupowane po p###_###)",
    )

    # 2) by_speaker
    bona_rows, (n_train, n_val, n_test) = build_bonafide_by_speaker(bonafide_dir, bonafide_files)
    rows = bona_rows + deepfake_rows
    save_and_summarize(
        rows,
        out_dir / "dataset_split_by_speaker.csv",
        "BY SPEAKER (bonafide split per mówca)",
    )
    print(f"Mówcy: {n_train} train / {n_val} val / {n_test} test")


if __name__ == "__main__":
    main()