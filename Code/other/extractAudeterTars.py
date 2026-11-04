"""
Rozpakowuje pliki tar pobrane z AUDETER do struktury:
    AudioFiles/Datasets/V2/raw/deepfake/<generator>/*.wav

Każdy generator ma dwa tary: dev/v0.tar i test/v0.tar — pliki z obu są
łączone do jednego folderu na generator. Ponieważ nazwy plików w dev/ i
test/ kolidują (te same ID), do nazwy dodawany jest prefiks "dev_" / "test_".
Skrypt jest idempotentny — istniejące pliki o tym samym rozmiarze są pomijane.
"""

import sys
import tarfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from projectMethods import get_audio_folder

SRC = get_audio_folder() / "AUDETER_audiobook" / "audiobook" / "tts"
DST = get_audio_folder() / "Datasets" / "V2" / "raw" / "deepfake"


def main():
    if not SRC.exists():
        print(f"Nie znaleziono źródła: {SRC}")
        return

    DST.mkdir(parents=True, exist_ok=True)
    print(f"Źródło: {SRC}")
    print(f"Cel:    {DST}\n")

    total_tars = 0
    total_extracted = 0
    total_skipped = 0

    for gen_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
        gen_name = gen_dir.name
        for split_dir in sorted(p for p in gen_dir.iterdir() if p.is_dir()):
            tar_path = split_dir / "v0.tar"
            if not tar_path.exists():
                continue
            out_dir = DST / gen_name
            out_dir.mkdir(parents=True, exist_ok=True)

            split_name = split_dir.name  # "dev" lub "test"
            extracted = skipped = 0
            with tarfile.open(tar_path, "r") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    # spłaszczamy: bierzemy samą nazwę pliku i dodajemy prefiks
                    # ze splitu, żeby uniknąć kolizji dev/ vs test/
                    leaf = Path(member.name).name
                    target = out_dir / f"{split_name}_{leaf}"
                    if target.exists() and target.stat().st_size == member.size:
                        skipped += 1
                        continue
                    with tar.extractfile(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    extracted += 1

            total_tars += 1
            total_extracted += extracted
            total_skipped += skipped
            print(f"  {gen_name:12s} / {split_dir.name:5s}: rozpakowano {extracted:5d}  "
                  f"(pominięto istniejących: {skipped})")

    print(f"\nRazem: {total_tars} tarów, {total_extracted} plików rozpakowanych, "
          f"{total_skipped} pominiętych.")


if __name__ == "__main__":
    main()