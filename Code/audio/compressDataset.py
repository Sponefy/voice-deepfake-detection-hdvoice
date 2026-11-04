"""
Kompresja HD Voice dla datasetu V2.
Przechodzi przez AudioFiles/Datasets/V2/raw/ i zapisuje wynik do
AudioFiles/Datasets/V2/compressed/, zachowując strukturę folderów.

Idempotentny — pliki już skompresowane są pomijane, więc można przerwać
i wznowić.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from audio.audioMethods import compress_audio_hd_voice_fast
from projectMethods import get_audio_folder

AUDIO_EXTS = {".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg"}
MAX_WORKERS = 12

SRC = get_audio_folder() / "Datasets" / "V2" / "raw"
DST = get_audio_folder() / "Datasets" / "V2" / "compressed"


def process_file(args):
    audio_file, src_root, dst_root = args
    rel = audio_file.relative_to(src_root)
    out_file = dst_root / rel.with_suffix(".wav")

    if out_file.exists() and out_file.stat().st_size > 0:
        return None  # już zrobione

    out_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        compress_audio_hd_voice_fast(str(audio_file), str(out_file))
        return None
    except Exception as e:
        return (rel, str(e))


def main():
    if not SRC.exists():
        print(f"Nie znaleziono źródła: {SRC}")
        return

    DST.mkdir(parents=True, exist_ok=True)
    print(f"Źródło: {SRC}")
    print(f"Cel:    {DST}\n")

    audio_files = [f for f in SRC.rglob("*") if f.suffix.lower() in AUDIO_EXTS]
    print(f"Znaleziono {len(audio_files)} plików audio.\n")

    tasks = [(f, SRC, DST) for f in audio_files]

    errors = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for res in tqdm(ex.map(process_file, tasks), total=len(tasks), desc="Kompresja"):
            if res is not None:
                errors.append(res)

    print("\n" + "=" * 60)
    print(f"Pomyślnie przetworzono: {len(audio_files) - len(errors)}/{len(audio_files)}")
    if errors:
        print(f"\nBłędy ({len(errors)}):")
        for path, msg in errors[:10]:
            print(f"  - {path}: {msg}")
        if len(errors) > 10:
            print(f"  ... i {len(errors) - 10} więcej")


if __name__ == "__main__":
    main()