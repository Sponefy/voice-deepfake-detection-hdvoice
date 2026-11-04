import re
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.append(str(Path(__file__).resolve().parents[1]))
from projectMethods import get_audio_folder

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

target_dir = get_audio_folder() / "In-The-Wild"
target_dir.mkdir(parents=True, exist_ok=True)


def download_peoples_speech_subset(
    out_dir: Path,
    budget_bytes: int = 15 * 1024 ** 3,        # ~15 GiB (mieści się w 10-20 GB)
    per_source_cap: int = 512 * 1024 ** 2,     # ~0.5 GiB na źródło -> różnorodność
    config: str = "clean",
) -> None:
    """Streamuje MLCommons/peoples_speech i zapisuje surowe FLAC-i do limitu rozmiaru.

    People's Speech to realna mowa ludzka (bonafide). Pobiera w trybie streaming
    (bez ściągania całego ~TB korpusu), z limitem łącznego rozmiaru oraz limitem
    na pojedyncze źródło (prefiks `id`), żeby wymusić różnorodność mówców/nagrań.

    Pliki układane są jako <out_dir>/<source>/<plik>.flac — dzięki czemu po
    kompresji HD Voice (compressDataset.py) split by_speaker traktuje każde
    źródło jak osobnego "mówcę" i nie ma leaku między train/val/test.
    """
    from datasets import Audio, load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    invalid = re.compile(r'[<>:"/\\|?*]')  # znaki nielegalne w nazwach plików Windows

    ds = load_dataset("MLCommons/peoples_speech", config, split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))  # surowe bajty FLAC, bez dekodowania

    total = 0
    per_source: dict[str, int] = {}
    n_files = 0
    # próg "budżet praktycznie wyczerpany" — gdy zostało mniej niż największy
    # spodziewany plik, kończymy zamiast bez końca pomijać duże próbki.
    min_remaining = 4 * 1024 ** 2  # 4 MiB

    manifest_path = out_dir / "manifest.tsv"
    with open(manifest_path, "w", encoding="utf-8", newline="") as manifest:
        manifest.write("filename\tsource\tduration_ms\n")

        for sample in ds:
            audio = sample.get("audio") or {}
            raw = audio.get("bytes")
            if not raw:
                continue

            # W People's Speech `id` koduje oryginalną ścieżkę: "/" -> "_SLASH_",
            # "." -> "_DOT_", i kończy się np. na ".flac". Źródłem (do limitu
            # różnorodności) jest pierwszy człon ścieżki, czyli część przed _SLASH_.
            sid = sample.get("id") or f"clip_{n_files}"
            if "_SLASH_" in sid:
                src, rest = sid.split("_SLASH_", 1)
            else:
                src, rest = "unknown", sid
            if per_source.get(src, 0) >= per_source_cap:
                continue  # to źródło już wyczerpane

            size = len(raw)
            remaining = budget_bytes - total
            if size > remaining:
                if remaining < min_remaining:
                    break          # budżet praktycznie wyczerpany
                continue           # ten plik za duży, ale szukaj mniejszych

            # bezpieczna nazwa + zachowane rozszerzenie audio
            ext = Path(audio.get("path") or sid).suffix or ".flac"
            fname = invalid.sub("_", rest) or f"clip_{n_files}"
            if not fname.endswith(ext):
                fname += ext

            dst_dir = out_dir / (invalid.sub("_", src) or "unknown")
            dst_dir.mkdir(parents=True, exist_ok=True)
            with open(dst_dir / fname, "wb") as f:
                f.write(raw)

            manifest.write(f"{src}/{fname}\t{src}\t{sample.get('duration_ms', '')}\n")
            total += size
            per_source[src] = per_source.get(src, 0) + size
            n_files += 1
            if n_files % 1000 == 0:
                print(f"  ... {n_files} plików, {total / 1024**3:.2f} GiB, {len(per_source)} źródeł")

    print(f"Pobrano {total / 1024**3:.2f} GiB w {n_files} plikach z {len(per_source)} źródeł")
    print(f"Manifest: {manifest_path}")

# snapshot_download(
#     repo_id="wqz995/AUDETER",
#     repo_type="dataset",
#     allow_patterns=[
#         "audiobook/tts/bark/*",
#         "audiobook/tts/cosyvoice/*",
#         "audiobook/tts/f5_tts/*",
#         "audiobook/tts/fish_speech/*",
#         "audiobook/tts/openvoice/*",
#         "audiobook/tts/sparktts/*",
#         "audiobook/tts/vits/*",
#         "audiobook/tts/xtts/*",
#         "audiobook/tts/yourtts/*",
#         "audiobook/tts/zonos/*",
#     ],
#     local_dir=str(target_dir),
# )

# LibriTTS-R train.clean.100 — bonafide z innego korpusu niż VCTK.
# Dane są w 18 shardach .parquet (~9.24 GB); audio jest zaszyte w kolumnie
# wewnątrz parquet — po pobraniu trzeba je wyekstrahować do plików .wav.
# snapshot_download(
#     repo_id="mythicinfinity/libritts_r",
#     repo_type="dataset",
#     allow_patterns=[
#         "data/train.clean.360/*.parquet",
#         "data/dev.clean/*.parquet",
#         "data/test.clean/*.parquet",
#     ],
#     local_dir=str(target_dir),
# )

# In-the-Wild (Müller et al. 2022) — bonafide + spoof z YouTube'a, używane
# jako benchmark out-of-distribution. Zawiera .wav i meta.csv (~10.7 GB).
# snapshot_download(
#     repo_id="mueller91/In-The-Wild",
#     repo_type="dataset",
#     local_dir=str(target_dir),
# )

# People's Speech (MLCommons) — realna, "zaszumiona" mowa ludzka (bonafide),
# duża różnorodność źródeł/mówców. Streaming z limitem rozmiaru i limitem na
# źródło. Trafia do V2/raw/bonafide/PeoplesSpeech -> podchwyci compressDataset.py.
download_peoples_speech_subset(
    out_dir=get_audio_folder() / "Datasets" / "V2" / "raw" / "bonafide" / "PeoplesSpeech",
    budget_bytes=15 * 1024 ** 3,
    per_source_cap=128 * 1024 ** 2,
)
