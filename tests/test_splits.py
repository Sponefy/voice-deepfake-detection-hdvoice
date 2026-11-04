import random
from pathlib import Path

import pytest

import projectMethods
from other.createBalancedOodSplit import derive_speaker, round_robin_by_duration


@pytest.mark.parametrize("relative, expected", [
    ("spk042/clip_001.wav", "spk042"),        # speaker folder
    ("p225/p225_001_mic1.wav", "p225"),        # VCTK folder
    ("p226_013_mic2.wav", "p226"),             # VCTK file without folder → regex
    ("lonely_file.wav", "lonely_file"),        # fallback → file stem
])
def test_derive_speaker(relative, expected):
    source = Path("/data/VCTK")
    assert derive_speaker(source / relative, source) == expected


def _files(speakers, per_speaker):
    return {s: [f"{s}_{i}.wav" for i in range(per_speaker)] for s in speakers}


def test_round_robin_spreads_cap_over_all_speakers():
    files = _files(["a", "b", "c", "d"], per_speaker=10)
    durations = {f: 1.0 for fs in files.values() for f in fs}

    picked, total = round_robin_by_duration(files, durations, cap_seconds=8, rng=random.Random(42))

    assert total == 8
    assert len(picked) == len(set(picked)) == 8
    per_speaker = {s: sum(f.startswith(s) for f in picked) for s in files}
    assert set(per_speaker.values()) == {2}  # 8 s over 4 speakers → 2 clips each


def test_round_robin_stops_when_data_runs_out():
    files = _files(["a", "b"], per_speaker=2)
    durations = {f: 1.0 for fs in files.values() for f in fs}

    picked, total = round_robin_by_duration(files, durations, cap_seconds=100, rng=random.Random(0))

    assert total == 4 and len(picked) == 4


def test_round_robin_is_reproducible_with_seed():
    files = _files(["a", "b", "c"], per_speaker=5)
    durations = {f: 1.0 for fs in files.values() for f in fs}

    run1, _ = round_robin_by_duration(files, durations, 7, random.Random(42))
    run2, _ = round_robin_by_duration(files, durations, 7, random.Random(42))

    assert run1 == run2


def test_project_root_defaults_to_repository(monkeypatch):
    monkeypatch.delenv("DEEPFAKE_PROJECT_ROOT", raising=False)
    root = projectMethods.find_project_root()
    assert (root / "Code" / "projectMethods.py").exists()


def test_project_root_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPFAKE_PROJECT_ROOT", str(tmp_path))
    assert projectMethods.find_project_root() == tmp_path.resolve()
    assert projectMethods.get_dataset_split_csv(5) == (
        tmp_path.resolve() / "AudioFiles" / "Datasets" / "V2" / "dataset_split_balanced_ood.csv"
    )


def test_unknown_dataset_version_raises():
    with pytest.raises(ValueError):
        projectMethods.get_dataset_split_csv(99)
