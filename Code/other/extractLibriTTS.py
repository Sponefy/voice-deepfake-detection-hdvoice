"""
Ekstrahuje audio z shardów parquet LibriTTS-R do plików .wav.

Wejście:  AudioFiles/LibriTTS-R/data/<split>/*.parquet
Wyjście:  AudioFiles/Datasets/V2/raw/bonafide/LibriTTS-R/<speaker_id>/<id>.wav

Audio jest zapisane w kolumnie audio.bytes jako gotowy WAV (RIFF) — wystarczy
zrzucić bajty do pliku, bez dekodowania. Pliki czytane są partiami
(iter_batches), żeby nie ładować całego shardu (~0.5 GB) do pamięci naraz.
Skrypt jest idempotentny — istniejące pliki są pomijane.
"""

import sys
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

# Wymuś UTF-8 na stdout (konsola Windows domyślnie cp1252 nie radzi sobie z "Ź" itp.)
sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from projectMethods import get_audio_folder

SRC = get_audio_folder() / "LibriTTS-R" / "data"
DST = get_audio_folder() / "Datasets" / "V2" / "raw" / "bonafide" / "LibriTTS-R"

BATCH_SIZE = 256


def main():
    if not SRC.exists():
        print(f"Nie znaleziono źródła: {SRC}")
        return

    parquet_files = sorted(SRC.rglob("*.parquet"))
    if not parquet_files:
        print(f"Brak plików .parquet w {SRC}")
        return

    DST.mkdir(parents=True, exist_ok=True)
    print(f"Źródło: {SRC}")
    print(f"Cel:    {DST}")
    print(f"Shardów: {len(parquet_files)}\n")

    total_written = 0
    total_skipped = 0

    for pf_path in parquet_files:
        pf = pq.ParquetFile(pf_path)
        n_rows = pf.metadata.num_rows
        written = skipped = 0

        desc = f"{pf_path.parent.name}/{pf_path.stem}"
        with tqdm(total=n_rows, desc=desc, leave=False) as bar:
            for batch in pf.iter_batches(
                batch_size=BATCH_SIZE, columns=["audio", "speaker_id", "id"]
            ):
                rows = batch.to_pylist()
                for r in rows:
                    speaker = r["speaker_id"]
                    file_id = r["id"]
                    audio_bytes = r["audio"]["bytes"]

                    out_dir = DST / speaker
                    out_file = out_dir / f"{file_id}.wav"

                    if out_file.exists() and out_file.stat().st_size == len(audio_bytes):
                        skipped += 1
                    else:
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_file.write_bytes(audio_bytes)
                        written += 1
                bar.update(len(rows))

        total_written += written
        total_skipped += skipped
        print(f"  {desc}: zapisano {written}, pominięto {skipped}")

    print("\n" + "=" * 60)
    print(f"Razem: zapisano {total_written}, pominięto {total_skipped}")


if __name__ == "__main__":
    main()