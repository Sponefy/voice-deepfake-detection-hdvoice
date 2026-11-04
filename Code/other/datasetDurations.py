import os
import soundfile as sf

base = os.path.join("AudioFiles", "Datasets", "V2", "raw", "deepfake")


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
    for f in os.listdir(folder):
        if not f.lower().endswith(".wav"):
            continue
        path = os.path.join(folder, f)
        try:
            info = sf.info(path)
            dur = info.duration
        except Exception:
            bad += 1
            continue
        total += dur
        count += 1
        mn = min(mn, dur)
        mx = max(mx, dur)
    avg = total / count if count else 0
    mn = 0 if count == 0 else mn
    rows.append((name, count, total, avg, mn, mx, bad))
    grand_total += total
    grand_count += count

print(f"{'Generator':<14}{'Pliki':>7}{'Suma':>16}{'Śr.':>9}{'Min':>8}{'Max':>8}")
print("-" * 62)
for name, count, total, avg, mn, mx, bad in rows:
    print(f"{name:<14}{count:>7}{fmt(total):>16}{avg:>7.2f}s{mn:>7.2f}s{mx:>7.2f}s"
          + (f"  (uszkodzone: {bad})" if bad else ""))
print("-" * 62)
print(f"{'RAZEM':<14}{grand_count:>7}{fmt(grand_total):>16}")