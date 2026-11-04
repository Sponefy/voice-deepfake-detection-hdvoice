"""
Sprawdza data leakage w dataset_split.csv:
- ta sama wypowiedź (utterance_id = p225_001) w różnych splitach przez mic1/mic2
- ten sam mówca w różnych splitach (info, nie zawsze problem)
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from projectMethods import get_dataset_split_csv

UTT_RE = re.compile(r"(p\d+)_(\d+)_mic[12]")


def main():
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = get_dataset_split_csv(5)  # domyślnie aktualny split (wersja 5)
    print(f"CSV: {csv_path}\n")

    utterance_splits = defaultdict(set)
    utterance_files = defaultdict(list)
    speaker_splits = defaultdict(set)

    bonafide_rows = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["label"] != "bonafide":
                continue
            bonafide_rows += 1
            fname = Path(row["file_path"]).stem
            m = UTT_RE.match(fname)
            if not m:
                continue
            speaker, utt = m.group(1), m.group(2)
            utt_id = f"{speaker}_{utt}"
            utterance_splits[utt_id].add(row["split"])
            utterance_files[utt_id].append((row["split"], fname))
            speaker_splits[speaker].add(row["split"])

    print(f"Bonafide wierszy: {bonafide_rows}")
    print(f"Unikalnych utterance_id: {len(utterance_splits)}")
    print(f"Unikalnych mówców: {len(speaker_splits)}\n")

    leaking = {u: splits for u, splits in utterance_splits.items() if len(splits) > 1}
    print(f"=== UTTERANCE-LEVEL LEAKAGE ===")
    print(f"Wypowiedzi w >1 splicie (mic1/mic2 rozjechane): {len(leaking)}")
    if leaking:
        pair_counts = defaultdict(int)
        for splits in leaking.values():
            pair_counts[tuple(sorted(splits))] += 1
        for pair, n in sorted(pair_counts.items()):
            print(f"  {pair}: {n}")
        print("\n  Przykłady (max 5):")
        for utt_id, splits in list(leaking.items())[:5]:
            print(f"    {utt_id}: {sorted(splits)}")
            for s, fn in utterance_files[utt_id]:
                print(f"        [{s:5s}] {fn}")

    print(f"\n=== SPEAKER-LEVEL ===")
    speakers_multi = {s: sp for s, sp in speaker_splits.items() if len(sp) > 1}
    print(f"Mówców w >1 splicie: {len(speakers_multi)} / {len(speaker_splits)}")
    if speakers_multi:
        only_in_one = len(speaker_splits) - len(speakers_multi)
        print(f"Mówców tylko w 1 splicie: {only_in_one}")
        # ilu mówców jest w train AND test/val
        in_train_and_test = sum(1 for sp in speaker_splits.values() if "train" in sp and "test" in sp)
        in_train_and_val = sum(1 for sp in speaker_splits.values() if "train" in sp and "val" in sp)
        print(f"Mówców jednocześnie w train i test: {in_train_and_test}")
        print(f"Mówców jednocześnie w train i val:  {in_train_and_val}")


if __name__ == "__main__":
    main()