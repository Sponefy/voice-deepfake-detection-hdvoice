"""
Wybiera podzbiór Mozilla Common Voice (en) jako nowe źródło bonafide dla datasetu V2.

Cel: zwiększyć RÓŻNORODNOŚĆ nagrań bonafide. Miarą ilości jest CZAS nagrań
(domyślnie >=150 h), a nie liczba plików ani GB.

Strategia różnorodności (round-robin po mówcach):
- bierze tylko klipy z validated.tsv (mowa zatwierdzona głosami społeczności),
- tylko te, które FAKTYCZNIE istnieją w clips/ (rozpakowywanie bywa przerwane),
- dobiera klipy rundami: w każdej rundzie po jednym klipie od każdego mówcy
  (w losowej kolejności), aż do osiągnięcia celu czasowego. Dzięki temu czas
  rozkłada się na maksymalną liczbę różnych mówców/akcentów, a nie na kilku
  gadatliwych. Limit klipów na mówcę (--max-per-speaker) ogranicza dominację.

Układ wyjścia (spójny z konwencją LibriTTS-R: mówca = pierwszy folder po źródle):
    AudioFiles/Datasets/V2/raw/bonafide/CommonVoice/spk00001/common_voice_en_*.mp3
Po uruchomieniu wystarczy puścić Twój mechanizm kompresji HD Voice z
source=...\\raw\\bonafide\\CommonVoice  ->  output=...\\compressed\\bonafide\\CommonVoice
(zachowa podfoldery spkNNNNN, więc split by_speaker będzie działać).

Manifest (CSV) zapisuje pełne client_id i metadane (czas, płeć, akcent, zdanie).

Uruchomienie:
    python Code/other/selectCommonVoiceSubset.py --dry-run     # tylko statystyki, nic nie kopiuje
    python Code/other/selectCommonVoiceSubset.py               # właściwa kopia (150 h)
    python Code/other/selectCommonVoiceSubset.py --target-hours 100 --max-per-speaker 8
"""

import argparse
import csv
import os
import random
import shutil
import sys
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from projectMethods import find_project_root

# --- Domyślna konfiguracja ---
DEFAULT_TARGET_HOURS = 150.0
DEFAULT_MAX_PER_SPEAKER = 10
DEFAULT_SEED = 42
# Zweryfikowana, częściowo rozpakowana kopia (1,4 mln klipów). Można nadpisać --cv-dir.
DEFAULT_CV_DIR = "AudioFiles/cv-corpus-25.0-2026-03-09/en"


def human_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-hours", type=float, default=DEFAULT_TARGET_HOURS,
                    help=f"Docelowy łączny czas nagrań w godzinach (domyślnie {DEFAULT_TARGET_HOURS})")
    ap.add_argument("--max-per-speaker", type=int, default=DEFAULT_MAX_PER_SPEAKER,
                    help=f"Maks. liczba klipów od jednego mówcy (domyślnie {DEFAULT_MAX_PER_SPEAKER})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--cv-dir", type=str, default=None,
                    help="Folder 'en' z Common Voice (zawiera validated.tsv, clip_durations.tsv, clips/). "
                         f"Domyślnie: {DEFAULT_CV_DIR}")
    ap.add_argument("--dry-run", action="store_true", help="Tylko statystyki + manifest, bez kopiowania audio")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = find_project_root()
    cv_dir = Path(args.cv_dir) if args.cv_dir else root / DEFAULT_CV_DIR
    clips_dir = cv_dir / "clips"
    validated_tsv = cv_dir / "validated.tsv"
    durations_tsv = cv_dir / "clip_durations.tsv"

    for p in (clips_dir, validated_tsv, durations_tsv):
        if not p.exists():
            sys.exit(f"❌ Brak: {p}")

    dst_root = root / "AudioFiles" / "Datasets" / "V2" / "raw" / "bonafide" / "CommonVoice"
    manifest_csv = root / "AudioFiles" / "Datasets" / "V2" / "commonvoice_selection.csv"
    target_seconds = args.target_hours * 3600.0

    print(f"Źródło CV:        {cv_dir}")
    print(f"Cel czasu:        {args.target_hours} h  ({target_seconds:.0f} s)")
    print(f"Limit/mówcę:      {args.max_per_speaker}   seed: {args.seed}")
    print(f"Wyjście (raw):    {dst_root}")
    print(f"Manifest:         {manifest_csv}\n")

    # 1) Zbiór plików faktycznie rozpakowanych w clips/
    print("[1/5] Skanuję clips/ (pliki faktycznie na dysku)...")
    existing = set()
    with os.scandir(clips_dir) as it:
        for e in it:
            if e.name.endswith(".mp3"):
                existing.add(e.name)
    print(f"      {len(existing):,} plików .mp3 obecnych na dysku")

    # 2) Czasy trwania klipów (clip -> sekundy)
    print("[2/5] Wczytuję clip_durations.tsv...")
    dur = pd.read_csv(durations_tsv, sep="\t", dtype={0: str}, usecols=[0, 1])
    dur.columns = ["clip", "ms"]
    dur["ms"] = pd.to_numeric(dur["ms"], errors="coerce")
    dur_map = dict(zip(dur["clip"], dur["ms"]))
    print(f"      {len(dur_map):,} wpisów czasu")

    # 3) validated.tsv -> kandydaci, którzy istnieją na dysku i mają znany czas
    print("[3/5] Wczytuję validated.tsv i filtruję kandydatów...")
    val = pd.read_csv(
        validated_tsv, sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
        usecols=["client_id", "path", "sentence", "age", "gender", "accents"],
        on_bad_lines="skip",
    )
    print(f"      {len(val):,} wierszy validated")

    by_speaker = defaultdict(list)  # client_id -> list[dict]
    n_missing_file = n_missing_dur = 0
    for r in val.itertuples(index=False):
        name = r.path
        if name not in existing:
            n_missing_file += 1
            continue
        ms = dur_map.get(name)
        if ms is None or ms <= 0:
            n_missing_dur += 1
            continue
        by_speaker[r.client_id].append({
            "client_id": r.client_id, "path": name, "dur_s": ms / 1000.0,
            "age": r.age, "gender": r.gender, "accents": r.accents, "sentence": r.sentence,
        })
    print(f"      kandydaci: {sum(len(v) for v in by_speaker.values()):,} klipów / "
          f"{len(by_speaker):,} mówców")
    print(f"      pominięto: {n_missing_file:,} (brak pliku na dysku), {n_missing_dur:,} (brak czasu)")

    # 4) Round-robin po mówcach aż do celu czasowego
    print("[4/5] Dobieram klipy (round-robin po mówcach)...")
    queues = {}
    for cid, clips in by_speaker.items():
        rng.shuffle(clips)
        queues[cid] = deque(clips[: args.max_per_speaker])
    speaker_order = list(queues.keys())
    rng.shuffle(speaker_order)

    selected = []
    total_s = 0.0
    progressing = True
    while total_s < target_seconds and progressing:
        progressing = False
        for cid in speaker_order:
            q = queues[cid]
            if q:
                selected.append(q.popleft())
                total_s += selected[-1]["dur_s"]
                progressing = True
                if total_s >= target_seconds:
                    break

    used_speakers = {c["client_id"] for c in selected}
    print(f"      wybrano {len(selected):,} klipów  |  {human_time(total_s)}  |  "
          f"{len(used_speakers):,} mówców")
    if total_s < target_seconds:
        print(f"      ⚠ Osiągnięto tylko {human_time(total_s)} z celu {args.target_hours} h "
              f"(wyczerpano dostępne klipy/mówców). Zmniejsz cel lub zwiększ --max-per-speaker.")

    # Statystyki różnorodności
    def dist(field):
        c = defaultdict(int)
        for s in selected:
            c[s[field] if (s[field] and str(s[field]) != "nan") else "(brak)"] += 1
        return dict(sorted(c.items(), key=lambda x: -x[1]))

    print("\n      Rozkład płci:", dist("gender"))
    acc = dist("accents")
    print(f"      Akcenty: {len(acc)} różnych, top:",
          dict(list(acc.items())[:6]))

    # Mapowanie client_id -> spkNNNNN (stabilne: kolejność pierwszego pojawienia)
    spk_id = {}
    for s in selected:
        if s["client_id"] not in spk_id:
            spk_id[s["client_id"]] = f"spk{len(spk_id) + 1:05d}"

    # 5) Manifest + kopiowanie
    print("\n[5/5] Zapis manifestu" + ("" if args.dry_run else " + kopiowanie plików") + "...")
    rows = []
    copied = 0
    for s in selected:
        spk = spk_id[s["client_id"]]
        dst = dst_root / spk / s["path"]
        rows.append({
            "file_path": str(dst), "speaker": spk, "client_id": s["client_id"],
            "duration_s": round(s["dur_s"], 3), "gender": s["gender"],
            "accents": s["accents"], "age": s["age"], "sentence": s["sentence"],
        })
        if not args.dry_run:
            src = clips_dir / s["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
            copied += 1
            if copied % 5000 == 0:
                print(f"      ... skopiowano {copied:,}/{len(selected):,}")

    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest_csv, index=False, encoding="utf-8")
    print(f"\n✓ Manifest: {manifest_csv}  ({len(rows):,} wierszy)")
    if args.dry_run:
        print("[DRY-RUN] Nie skopiowano żadnych plików.")
    else:
        print(f"✓ Skopiowano {copied:,} plików do {dst_root}")
        print("\nDalej: uruchom kompresję HD Voice z:")
        print(f"   source = {dst_root}")
        print(f"   output = {root / 'AudioFiles' / 'Datasets' / 'V2' / 'compressed' / 'bonafide' / 'CommonVoice'}")


if __name__ == "__main__":
    main()