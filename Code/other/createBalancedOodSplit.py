"""
Buduje split datasetu V2 z BALANSEM źródeł bonafide PO CZASIE i ewaluacją OOD.

Motywacja: bonafide jest teraz znacznie więcej niż deepfake (738h vs 308h) i jest
zdominowane przez pojedyncze źródła. Bez balansu model uczy się rozpoznawać
DOMENĘ konkretnego źródła zamiast samej syntezy (shortcut learning) — patrz
docs/experiments_pl.md.

Ten skrypt (na podstawie createCleanSplits.py + datasetDurationsBonafide.py):
1. BONAFIDE — bierze MNIEJ danych, mniej więcej PO RÓWNO (wg ŁĄCZNEGO CZASU
   nagrań, nie liczby plików) z każdego źródła (CommonVoice / LibriTTS-R /
   PeoplesSpeech / VCTK_Corpus). Łączny czas bonafide ≈ czas deepfake (--ratio).
   W obrębie źródła:
   - split PO MÓWCY (mówca = pierwszy folder po nazwie źródła) → zero leaku,
   - dobór round-robin po mówcach do limitu CZASU → maks. różnorodność mówców.
2. DEEPFAKE — bierze wszystko, split po pliku w obrębie każdego źródła
   (zachowuje proporcje generatorów). Tylko train/val.
3. TEST = OOD — In-the-Wild z folderu compressed/test/, etykiety z meta.csv
   (bona-fide → bonafide, spoof → deepfake). Jedyna uczciwa miara generalizacji.

Wynik: jeden CSV (file_path, label, split) ze splitami train/val/test, gdzie
`test` to OOD. Pasuje bez zmian do AudioDatasetFromCSV i train.py.

Uruchomienie:
    python Code/other/createBalancedOodSplit.py                  # ratio 1.0 (czas bonafide ≈ deepfake)
    python Code/other/createBalancedOodSplit.py --ratio 1.5      # 1.5× więcej czasu bonafide
    python Code/other/createBalancedOodSplit.py --per-source-hours 60
    python Code/other/createBalancedOodSplit.py --dry-run        # tylko statystyki
"""

import argparse
import random
import re
import sys
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))
from projectMethods import get_dataset_path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

AUDIO_EXTS = {".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg", ".opus"}
UTT_RE = re.compile(r"(p\d+)_(\d+)_mic[12]")  # VCTK: pXXX_XXX_micY

TRAIN_RATIO = 0.85   # in-domain dzielimy tylko na train/val (test = OOD)
VAL_RATIO = 0.15
RANDOM_STATE = 42
WORKERS = 16

OUT_NAME = "dataset_split_balanced_ood.csv"


def collect_audio(folder: Path):
    # sorted → powtarzalność (rglob nie gwarantuje stałej kolejności między uruchomieniami)
    return sorted(f for f in folder.rglob("*") if f.suffix.lower() in AUDIO_EXTS)


def duration(path: Path) -> float:
    try:
        return sf.info(str(path)).duration
    except Exception:
        return 0.0


def compute_durations(files):
    """{Path: sekundy} równolegle (jak datasetDurationsBonafide.py, ThreadPool)."""
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        secs = list(ex.map(duration, files))
    return dict(zip(files, secs))


def fmt_h(seconds: float) -> str:
    return f"{seconds / 3600:.1f}h"


def derive_speaker(f: Path, source_dir: Path) -> str:
    """Mówca = pierwszy folder po nazwie źródła. VCTK (pliki pXXX_..._micY w
    folderze pXXX) i tak da pXXX. Fallback: regex VCTK na nazwie, potem stem."""
    rel = f.relative_to(source_dir)
    if len(rel.parts) > 1:
        return rel.parts[0]
    m = UTT_RE.match(f.stem)
    return m.group(1) if m else f.stem


def round_robin_by_duration(files_by_speaker: dict, dur_map: dict, cap_seconds: float,
                            rng: random.Random):
    """Dobiera pliki do limitu CZASU, w każdej rundzie po jednym od każdego mówcy
    (losowa kolejność) → rozkłada limit na maks. liczbę różnych mówców."""
    queues = {}
    for spk in sorted(files_by_speaker):
        fs = sorted(files_by_speaker[spk])   # stała baza przed shuffle → powtarzalność
        rng.shuffle(fs)
        queues[spk] = deque(fs)
    order = sorted(queues.keys())
    rng.shuffle(order)

    picked = []
    total = 0.0
    progressing = True
    while total < cap_seconds and progressing:
        progressing = False
        for spk in order:
            q = queues[spk]
            if q:
                f = q.popleft()
                picked.append(f)
                total += dur_map.get(f, 0.0)
                progressing = True
                if total >= cap_seconds:
                    break
    return picked, total


def build_bonafide_balanced(bonafide_dir: Path, per_source_seconds: float, rng: random.Random):
    """Dla każdego źródła: split po mówcy, round-robin do limitu czasu (train+val)."""
    rows = []
    stats = []
    sources = sorted(p for p in bonafide_dir.iterdir() if p.is_dir())

    for src_dir in sources:
        files = collect_audio(src_dir)
        if not files:
            continue
        dur_map = compute_durations(files)
        avail = sum(dur_map.values())

        by_speaker = defaultdict(list)
        for f in files:
            by_speaker[derive_speaker(f, src_dir)].append(f)
        speakers = sorted(by_speaker)

        # split MÓWCÓW na train/val (zero leaku mówcy między splitami)
        if len(speakers) >= 2:
            train_spk, val_spk = train_test_split(
                speakers, test_size=VAL_RATIO, random_state=RANDOM_STATE
            )
        else:
            train_spk, val_spk = speakers, []
        train_spk, val_spk = set(train_spk), set(val_spk)

        # limit czasu nie większy niż dostępny w źródle
        cap = min(per_source_seconds, avail)
        train_cap = cap * TRAIN_RATIO
        val_cap = cap * VAL_RATIO

        train_pool = {s: by_speaker[s] for s in train_spk}
        val_pool = {s: by_speaker[s] for s in val_spk}

        train_sel, train_dur = round_robin_by_duration(train_pool, dur_map, train_cap, rng)
        val_sel, val_dur = round_robin_by_duration(val_pool, dur_map, val_cap, rng)

        for f in train_sel:
            rows.append({"file_path": str(f), "label": "bonafide", "split": "train"})
        for f in val_sel:
            rows.append({"file_path": str(f), "label": "bonafide", "split": "val"})

        stats.append((src_dir.name, avail, len(speakers), len(train_sel), train_dur,
                      len(val_sel), val_dur))

    print("\n=== BONAFIDE (balans po CZASIE, split po mówcy) ===")
    print(f"{'źródło':<16}{'dostępne':>10}{'mówców':>8}{'plik_tr':>9}{'czas_tr':>9}{'plik_val':>9}{'czas_val':>9}")
    for s in stats:
        print(f"{s[0]:<16}{fmt_h(s[1]):>10}{s[2]:>8}{s[3]:>9}{fmt_h(s[4]):>9}{s[5]:>9}{fmt_h(s[6]):>9}")
    return rows


def build_deepfake(deepfake_dir: Path, seed: int):
    """Wszystkie deepfake, split po pliku w obrębie każdego źródła (train/val)."""
    rows = []
    stats = []
    for src_dir in sorted(p for p in deepfake_dir.iterdir() if p.is_dir()):
        files = collect_audio(src_dir)
        if not files:
            stats.append((src_dir.name, 0, 0, 0))
            continue
        train_f, val_f = train_test_split(files, test_size=VAL_RATIO, random_state=seed)
        for f in train_f:
            rows.append({"file_path": str(f), "label": "deepfake", "split": "train"})
        for f in val_f:
            rows.append({"file_path": str(f), "label": "deepfake", "split": "val"})
        stats.append((src_dir.name, len(files), len(train_f), len(val_f)))

    print("\n=== DEEPFAKE (wszystko, split po pliku per źródło) ===")
    print(f"{'źródło':<16}{'plików':>9}{'train':>9}{'val':>9}")
    for s in stats:
        print(f"{s[0]:<16}{s[1]:>9}{s[2]:>9}{s[3]:>9}")
    return rows


def build_ood_test(compressed_dir: Path, ood_val_frac: float, seed: int):
    """OOD In-the-Wild dzielone PO MÓWCY na dwa rozłączne splity (bez leaku):
       - val_ood : kalibracja progu / uczciwa selekcja modelu,
       - test    : finalny, nietykany pomiar OOD.
    Etykiety z meta.csv (raw/), audio z compressed/test/ (ten sam HD Voice)."""
    comp_test = compressed_dir / "test"
    # audio: compressed/test/In-The-Wild (pliki 0.wav, 1.wav, ...)
    audio_base = None
    if comp_test.exists():
        for d in [comp_test] + [p for p in comp_test.iterdir() if p.is_dir()]:
            if any(d.glob("*.wav")):
                audio_base = d
                break
    # meta.csv szukaj w raw/test (kompresja nie kopiuje csv) i compressed/test
    meta = None
    for root in (compressed_dir.parent / "raw" / "test", comp_test):
        hits = list(root.rglob("meta.csv")) if root.exists() else []
        if hits:
            meta = hits[0]
            break
    if meta is None or audio_base is None:
        print("\n⚠ OOD: nie znaleziono meta.csv lub audio In-the-Wild — pomijam split test.")
        return []

    df = pd.read_csv(meta)
    label_map = {"bona-fide": "bonafide", "bonafide": "bonafide",
                 "spoof": "deepfake", "deepfake": "deepfake"}
    entries = []   # (wav, label, speaker)
    missing = 0
    for r in df.itertuples(index=False):
        lbl = label_map.get(str(r.label).strip().lower())
        if lbl is None:
            continue
        wav = audio_base / (Path(str(r.file)).stem + ".wav")
        if not wav.exists():
            missing += 1
            continue
        entries.append((wav, lbl, str(r.speaker)))

    # split MÓWCÓW na val_ood/test (mówca = osoba; In-the-Wild ma realne i fake
    # tej samej osoby → split po mówcy nie pozwala tożsamości wyciec między splity)
    speakers = sorted({e[2] for e in entries})
    if len(speakers) >= 2 and 0 < ood_val_frac < 1:
        val_spk, _ = train_test_split(speakers, train_size=ood_val_frac, random_state=seed)
        val_spk = set(val_spk)
    else:
        val_spk = set()

    rows = []
    for wav, lbl, spk in entries:
        rows.append({"file_path": str(wav), "label": lbl,
                     "split": "val_ood" if spk in val_spk else "test"})

    def counts(split):
        sub = [x for x in rows if x["split"] == split]
        b = sum(1 for x in sub if x["label"] == "bonafide")
        return len(sub), b, len(sub) - b

    nv, nvb, nvs = counts("val_ood")
    nt, ntb, nts = counts("test")
    print("\n=== OOD In-the-Wild (podział po mówcy) ===")
    print(f"meta.csv: {meta}")
    print(f"audio:    {audio_base}")
    print(f"mówcy: {len(speakers)} → val_ood {len(val_spk)} / test {len(speakers) - len(val_spk)}")
    print(f"val_ood: {nv:>6} (bonafide={nvb}, deepfake={nvs})   [kalibracja progu]")
    print(f"test:    {nt:>6} (bonafide={ntb}, deepfake={nts})   [finalny OOD]")
    print(f"brak na dysku: {missing}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="Łączny CZAS bonafide = ratio × czas deepfake (domyślnie 1.0)")
    ap.add_argument("--per-source-hours", type=float, default=None,
                    help="Sztywny limit GODZIN bonafide na źródło (nadpisuje --ratio)")
    ap.add_argument("--ood-val-frac", type=float, default=0.5,
                    help="Ułamek MÓWCÓW In-the-Wild do val_ood (kalibracja); reszta → test (domyślnie 0.5)")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    ap.add_argument("--dry-run", action="store_true", help="Tylko statystyki, nie zapisuje CSV")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    compressed_dir = get_dataset_path()              # .../V2/compressed
    out_path = compressed_dir.parent / OUT_NAME       # .../V2/dataset_split_balanced_ood.csv
    bonafide_dir = compressed_dir / "bonafide"
    deepfake_dir = compressed_dir / "deepfake"

    # czas deepfake → cel czasu bonafide
    deepfake_files = collect_audio(deepfake_dir)
    print(f"Liczę czas deepfake ({len(deepfake_files)} plików)...")
    deepfake_seconds = sum(compute_durations(deepfake_files).values())
    n_bona_sources = sum(1 for p in bonafide_dir.iterdir() if p.is_dir() and any(collect_audio(p)))

    if args.per_source_hours is not None:
        per_source_seconds = args.per_source_hours * 3600
    else:
        per_source_seconds = deepfake_seconds * args.ratio / max(1, n_bona_sources)

    print(f"Deepfake (wszystkie): {fmt_h(deepfake_seconds)}")
    print(f"Źródła bonafide:      {n_bona_sources}")
    print(f"Cel bonafide/źródło:  {fmt_h(per_source_seconds)}  "
          f"(≈ {fmt_h(per_source_seconds * n_bona_sources)} łącznie, ratio={args.ratio})")

    rows = []
    rows += build_bonafide_balanced(bonafide_dir, per_source_seconds, rng)
    rows += build_deepfake(deepfake_dir, args.seed)
    rows += build_ood_test(compressed_dir, args.ood_val_frac, args.seed)

    df = pd.DataFrame(rows)
    print("\n=== PODSUMOWANIE (split × label, liczba plików) ===")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))

    if args.dry_run:
        print("\n[DRY-RUN] CSV nie zapisany.")
        return
    df.to_csv(out_path, index=False)
    print(f"\n✓ Zapisano: {out_path}  ({len(df)} wierszy)")


if __name__ == "__main__":
    main()