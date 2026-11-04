"""
Audyt CSV-a dataset_split_by_speaker.csv:
1. Duplikaty ścieżek między splitami
2. Rozkład bonafide po źródle (AUDETER / LibriTTS-R) i splicie
3. Przeciek mówców między splitami (główna obietnica by_speaker)
4. Kolizja namespace mówców AUDETER vs LibriTTS-R
5. Rozkład generatorów deepfake per split
6. Pokrycie regex UTT_RE na AUDETER
7. Sanity check ID mówców LibriTTS-R
"""

import re
from pathlib import Path

import pandas as pd

CSV = Path("AudioFiles/Datasets/V2/dataset_split_by_speaker.csv")
df = pd.read_csv(CSV)
print(f"CSV: {CSV}")
print(f"rows: {len(df)}")
print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
print()

UTT_RE = re.compile(r"(p\d+)_(\d+)_mic[12]")


def norm(path: str) -> str:
    return path.replace("\\", "/")


def extract_speaker(file_path: str, label: str):
    if label != "bonafide":
        return None
    p = Path(file_path)
    m = UTT_RE.match(p.stem)
    if m:
        return m.group(1)            # AUDETER: p225_001_mic1 -> p225
    return p.parent.name             # LibriTTS-R: 100/100_121669_*.wav -> 100


def bona_source(path: str) -> str:
    s = norm(path)
    if "AUDETER" in s:
        return "AUDETER"
    if "LibriTTS-R" in s:
        return "LibriTTS-R"
    return "OTHER"


def deepfake_gen(path: str) -> str:
    parts = norm(path).split("/")
    for i, p in enumerate(parts):
        if p == "deepfake" and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


df["speaker"] = df.apply(lambda r: extract_speaker(r["file_path"], r["label"]), axis=1)
df.loc[df["label"] == "bonafide", "source"] = (
    df.loc[df["label"] == "bonafide", "file_path"].map(bona_source)
)
df.loc[df["label"] == "deepfake", "generator"] = (
    df.loc[df["label"] == "deepfake", "file_path"].map(deepfake_gen)
)

# === 1 ===
print("=" * 60)
print("1. DUPLIKATY ŚCIEŻEK")
print("=" * 60)
dup_mask = df["file_path"].duplicated(keep=False)
print(f"Duplikatów: {dup_mask.sum()}")
if dup_mask.sum() > 0:
    print(df[dup_mask].sort_values("file_path").head(20))
print()

# === 2 ===
print("=" * 60)
print("2. BONAFIDE — rozkład po źródle × split")
print("=" * 60)
bonafide = df[df["label"] == "bonafide"]
print(bonafide.groupby(["source", "split"]).size().unstack(fill_value=0))
print()
print("Mówcy unikatowi per source:")
for src in ["AUDETER", "LibriTTS-R"]:
    s = bonafide[bonafide["source"] == src]
    print(f"  {src:<12}: {s['speaker'].nunique():>5} mówców, {len(s):>7} plików")
print()

# === 3 ===
print("=" * 60)
print("3. PRZECIEK MÓWCÓW BONAFIDE między splitami")
print("=" * 60)
speakers_per_split = bonafide.groupby("speaker")["split"].nunique()
leaking = speakers_per_split[speakers_per_split > 1]
print(f"Mówców w >1 splicie: {len(leaking)}")
if len(leaking) > 0:
    print("Przykłady:")
    print(leaking.head(10).to_string())
    # podział tych przeciekających
    bad = bonafide[bonafide["speaker"].isin(leaking.index)]
    print()
    print("Tabela rozkładu (mówca × split):")
    print(bad.groupby(["speaker", "split"]).size().unstack(fill_value=0).head(20))
else:
    print("✓ Brak przecieku — każdy mówca w dokładnie jednym splicie.")
print()

# === 4 ===
print("=" * 60)
print("4. KOLIZJA NAMESPACE: AUDETER (p###) vs LibriTTS-R (###)")
print("=" * 60)
audeter_speakers = set(bonafide[bonafide["source"] == "AUDETER"]["speaker"])
libri_speakers = set(bonafide[bonafide["source"] == "LibriTTS-R"]["speaker"])
collision = audeter_speakers & libri_speakers
print(f"AUDETER:    {len(audeter_speakers):>5} mówców (przykład: {sorted(audeter_speakers)[:3]})")
print(f"LibriTTS-R: {len(libri_speakers):>5} mówców (przykład: {sorted(libri_speakers)[:3]})")
print(f"Wspólne ID: {len(collision)}")
if collision:
    print("Kolidujące:", sorted(collision)[:20])
print()

# === 5 ===
print("=" * 60)
print("5. DEEPFAKE — generatory × split")
print("=" * 60)
deepfake = df[df["label"] == "deepfake"]
table = deepfake.groupby(["generator", "split"]).size().unstack(fill_value=0)
table["total"] = table.sum(axis=1)
print(table)
print()
# Czy każdy generator ma proporcje ~70/15/15?
print("Frakcje per generator:")
frac = deepfake.groupby(["generator", "split"]).size().unstack(fill_value=0)
frac_norm = frac.div(frac.sum(axis=1), axis=0).round(3)
print(frac_norm)
print()

# === 6 ===
print("=" * 60)
print("6. POKRYCIE REGEX UTT_RE na AUDETER")
print("=" * 60)
audeter = bonafide[bonafide["source"] == "AUDETER"]
matched = audeter["file_path"].apply(lambda p: bool(UTT_RE.match(Path(p).stem)))
print(f"AUDETER plików: {len(audeter)}")
print(f"Pasujących regex 'p###_###_mic[12]': {matched.sum()}")
print(f"NIE pasujących: {(~matched).sum()}")
if (~matched).sum() > 0:
    print("Przykłady niezgodne:")
    for p in audeter[~matched]["file_path"].head(5).tolist():
        print(f"  {Path(p).name}")
print()

# === 7 ===
print("=" * 60)
print("7. LibriTTS-R — sanity check ID mówców")
print("=" * 60)
libri = bonafide[bonafide["source"] == "LibriTTS-R"].head(10)
print("Przykłady (speaker_id z CSV / parent_dir / file):")
for _, r in libri.iterrows():
    p = Path(r["file_path"])
    print(f"  speaker={r['speaker']:>6}  parent={p.parent.name:>6}  file={p.name}")
print()

# === 8 ===
print("=" * 60)
print("8. PODSUMOWANIE LICZB W CSV")
print("=" * 60)
print(f"Wszystkich rekordów:     {len(df):>7}")
print(f"  bonafide:               {len(bonafide):>7}")
print(f"    AUDETER:              {(bonafide['source']=='AUDETER').sum():>7}")
print(f"    LibriTTS-R:           {(bonafide['source']=='LibriTTS-R').sum():>7}")
print(f"  deepfake:               {len(deepfake):>7}")
print(f"    unikatowych generatorów: {deepfake['generator'].nunique():>4}")