import os
import soundfile as sf

base = os.path.join("AudioFiles", "Datasets", "V2", "raw", "bonafide")
AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3", ".opus", ".m4a")


def fmt(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h}h {m}m {s:05.2f}s"
    return f"{m}m {s:05.2f}s"


grand_total = 0.0
grand_count = 0
rows = []

for name in sorted(os.listdir(base)):
    folder = os.path.join(base, name)
    if not os.path.isdir(folder):
        continue
    total = 0.0
    count = 0
    mn = float("inf")
    mx = 0.0
    bad = 0
    for root, _, files in os.walk(folder):
        for f in files:
            if not f.lower().endswith(AUDIO_EXT):
                continue
            path = os.path.join(root, f)
            try:
                dur = sf.info(path).duration
            except Exception:
                bad += 1
                continue
            total += dur
            count += 1
            if dur < mn:
                mn = dur
            if dur > mx:
                mx = dur
    if count == 0:
        mn = 0
    rows.append((name, count, total, total / count if count else 0, mn, mx, bad))
    grand_total += total
    grand_count += count

print(f"{'Źródło':<14}{'Pliki':>9}{'Suma':>18}{'Śr.':>10}{'Min':>9}{'Max':>9}")
print("-" * 70)
for name, count, total, avg, mn, mx, bad in rows:
    line = f"{name:<14}{count:>9}{fmt(total):>18}{avg:>8.2f}s{mn:>8.2f}s{mx:>8.2f}s"
    if bad:
        line += f"  (uszkodzone: {bad})"
    print(line)
print("-" * 70)
print(f"{'RAZEM':<14}{grand_count:>9}{fmt(grand_total):>18}")