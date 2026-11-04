"""
Sprawdza rozmiar i długość datasetu V1.
Wypisuje:
- całkowity rozmiar (GB) i sumaryczny czas nagrań dla bonafide i deepfake
- rozmiar i czas dla każdego podkatalogu w deepfake
"""

import sys
from pathlib import Path

import soundfile as sf

sys.path.append(str(Path(__file__).resolve().parents[1]))
from projectMethods import get_dataset_path


def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}h {m:02d}m {s:05.2f}s"


def scan_folder(folder: Path):
    """Zwraca (liczba_plików, suma_bajtów, suma_sekund) dla wszystkich .wav w folderze (rekursywnie)."""
    total_bytes = 0
    total_seconds = 0.0
    count = 0
    for wav in folder.rglob("*.wav"):
        try:
            info = sf.info(str(wav))
            total_seconds += info.frames / info.samplerate
            total_bytes += wav.stat().st_size
            count += 1
        except Exception as e:
            print(f"  ! Pominięto {wav.name}: {e}")
    return count, total_bytes, total_seconds


def print_row(label: str, count: int, size_bytes: int, seconds: float):
    gb = size_bytes / (1024 ** 3)
    print(f"  {label:<20} {count:>8d} plików   {gb:>8.3f} GB   {format_duration(seconds)}")


def main():
    dataset_path = get_dataset_path()
    print(f"Dataset: {dataset_path}\n")

    if not dataset_path.exists():
        print("Katalog datasetu nie istnieje.")
        return

    summary = {}
    for cls in ("bonafide", "deepfake"):
        cls_path = dataset_path / cls
        if not cls_path.exists():
            print(f"Brak katalogu: {cls_path}")
            continue

        print(f"=== {cls.upper()} ===")

        if cls == "deepfake":
            subdirs = sorted([d for d in cls_path.iterdir() if d.is_dir()])
            cls_count = cls_bytes = 0
            cls_seconds = 0.0
            for sub in subdirs:
                count, size_bytes, seconds = scan_folder(sub)
                print_row(sub.name, count, size_bytes, seconds)
                cls_count += count
                cls_bytes += size_bytes
                cls_seconds += seconds
        else:
            cls_count, cls_bytes, cls_seconds = scan_folder(cls_path)

        print("  " + "-" * 70)
        print_row("RAZEM " + cls, cls_count, cls_bytes, cls_seconds)
        print()
        summary[cls] = (cls_count, cls_bytes, cls_seconds)

    print("=== PODSUMOWANIE ===")
    total_bytes = sum(v[1] for v in summary.values())
    total_seconds = sum(v[2] for v in summary.values())
    total_count = sum(v[0] for v in summary.values())
    for cls, (c, b, s) in summary.items():
        print_row(cls, c, b, s)
    print("  " + "-" * 70)
    print_row("ŁĄCZNIE", total_count, total_bytes, total_seconds)


if __name__ == "__main__":
    main()