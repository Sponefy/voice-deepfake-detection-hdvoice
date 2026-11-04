"""
Tworzy mniejszy podzbiór datasetu V2 (~7 GB z ~72 GB).

Strategia (speaker-level, stratyfikacja po (klasa, źródło, split)):
- bonafide/AUDETER:    wybiera FRACTION mówców z każdego splitu, kopiuje wszystkie ich pliki
- bonafide/LibriTTS-R: jw., mówca = nazwa folderu nadrzędnego
- deepfake/<gen>:      brak mówców w strukturze plików — wybiera FRACTION plików losowo
                       z każdego (generator, split), proporcje generatorów zachowane

Wejście:  AudioFiles/Datasets/V2/dataset_split_by_speaker.csv  (+ pliki audio)
Wyjście:  AudioFiles/Datasets/V2_small/compressed/...
          AudioFiles/Datasets/V2_small/dataset_split_by_speaker.csv

Uruchomienie:
    python Code/other/createSmallSubset.py            # rzeczywista kopia
    python Code/other/createSmallSubset.py --dry-run  # tylko statystyki
"""

import argparse
import random
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from projectMethods import find_project_root

FRACTION = 0.10
RANDOM_SEED = 42
UTT_RE = re.compile(r"(p\d+)_(\d+)_mic[12]")


def derive_speaker(rel_path: Path, source: str) -> str:
    """Wyciąga identyfikator mówcy z bonafide relpath (compressed/bonafide/<source>/...)."""
    if source == "AUDETER":
        m = UTT_RE.match(rel_path.stem)
        return m.group(1) if m else rel_path.parent.name
    # LibriTTS-R i inne — mówca = pierwszy folder po nazwie źródła
    parts = rel_path.parts
    return parts[parts.index(source) + 1] if source in parts else rel_path.parent.name


def normalize(p: str) -> Path:
    return Path(p.replace("\\", "/"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, default=FRACTION)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    root = find_project_root()
    src_root = root / "AudioFiles" / "Datasets" / "V2"
    dst_root = root / "AudioFiles" / "Datasets" / "V2_small"
    csv_in = src_root / "dataset_split_by_speaker.csv"

    df = pd.read_csv(csv_in)
    print(f"Wczytano {len(df)} wierszy z {csv_in.name}")
    print(f"Frakcja: {args.fraction:.0%}   seed: {args.seed}\n")

    # Wyciągnij (cls, source, rel_path) dla każdego wiersza
    pat = re.compile(r".*compressed[\\/](bonafide|deepfake)[\\/]([^\\/]+)[\\/](.+)$")
    parsed = []
    for fp in df["file_path"]:
        m = pat.match(fp)
        if not m:
            raise RuntimeError(f"Nieparsowalna ścieżka: {fp}")
        cls, source, rel = m.group(1), m.group(2), m.group(3)
        parsed.append((cls, source, Path(rel)))
    df["cls"] = [t[0] for t in parsed]
    df["source"] = [t[1] for t in parsed]
    df["rel"] = [t[2] for t in parsed]

    selected_rows = []

    # --- BONAFIDE: speaker-level w obrębie (source, split) ---
    bona = df[df["cls"] == "bonafide"].copy()
    bona["speaker"] = [derive_speaker(r, s) for r, s in zip(bona["rel"], bona["source"])]

    print("=== BONAFIDE (per source × split) ===")
    print(f"{'source':<14}{'split':<8}{'speakers':>10}{'picked':>10}{'files_in':>12}{'files_out':>12}")
    for (source, split), grp in bona.groupby(["source", "split"]):
        speakers = sorted(grp["speaker"].unique())
        n_pick = max(1, round(len(speakers) * args.fraction))
        picked = set(rng.sample(speakers, n_pick))
        sel = grp[grp["speaker"].isin(picked)]
        print(f"{source:<14}{split:<8}{len(speakers):>10}{n_pick:>10}{len(grp):>12}{len(sel):>12}")
        selected_rows.append(sel)

    # --- DEEPFAKE: per (generator, split), wybór plikowy ---
    deep = df[df["cls"] == "deepfake"].copy()
    print("\n=== DEEPFAKE (per generator × split, sampling plikowy) ===")
    print(f"{'generator':<14}{'split':<8}{'files_in':>12}{'files_out':>12}")
    for (source, split), grp in deep.groupby(["source", "split"]):
        n_pick = max(1, round(len(grp) * args.fraction))
        indices = rng.sample(list(grp.index), n_pick)
        sel = grp.loc[indices]
        print(f"{source:<14}{split:<8}{len(grp):>12}{len(sel):>12}")
        selected_rows.append(sel)

    out = pd.concat(selected_rows, ignore_index=True)
    print(f"\nŁącznie wybrano {len(out)} plików z {len(df)} ({len(out)/len(df):.1%})")
    print(out.groupby(["split", "label"]).size().unstack(fill_value=0))

    # --- KOPIOWANIE PLIKÓW + BUDOWANIE NOWYCH ŚCIEŻEK ---
    new_paths = []
    total_bytes = 0
    n_copied = 0
    n_skipped = 0
    for _, row in out.iterrows():
        rel_under_compressed = Path(row["cls"]) / row["source"] / row["rel"]
        src_file = src_root / "compressed" / rel_under_compressed
        dst_file = dst_root / "compressed" / rel_under_compressed

        if not src_file.exists():
            # Spróbuj absolutnej ścieżki z CSV jako fallback
            src_file = Path(row["file_path"])
        if not src_file.exists():
            n_skipped += 1
            new_paths.append(None)
            continue

        new_paths.append(str(dst_file))
        if args.dry_run:
            total_bytes += src_file.stat().st_size
            continue

        dst_file.parent.mkdir(parents=True, exist_ok=True)
        if not dst_file.exists():
            shutil.copy2(src_file, dst_file)
        total_bytes += dst_file.stat().st_size
        n_copied += 1
        if n_copied % 2000 == 0:
            print(f"  ... skopiowano {n_copied} plików ({total_bytes/1024**3:.2f} GB)")

    out["file_path"] = new_paths
    out = out[out["file_path"].notna()][["file_path", "label", "split"]]

    print(f"\nRozmiar zbioru: {total_bytes/1024**3:.2f} GB")
    if n_skipped:
        print(f"UWAGA: pominięto {n_skipped} plików (nie istniały na dysku)")

    if args.dry_run:
        print("\n[DRY-RUN] nic nie zostało zapisane.")
        return

    csv_out = dst_root / "dataset_split_by_speaker.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_out, index=False)
    print(f"\nZapisano CSV: {csv_out}")
    print(f"Skopiowano plików: {n_copied}")


if __name__ == "__main__":
    main()