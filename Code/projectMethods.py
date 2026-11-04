import os
from pathlib import Path
import torch

# Wyświetlenie informacji o urządzeniu
def get_device(verbose=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print("CUDA GPU:", torch.cuda.get_device_name(0))
            print("CUDA Memory:", f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")    
        if verbose:
            print("Mac Metal Performance Shaders (MPS) - Apple Silicon GPU")
    else:
        device = torch.device("cpu")
        if verbose:
            print("CPU - brak akceleracji GPU")
    return device

# Katalog główny projektu (dane w <root>/AudioFiles, modele w <root>/Models)
def find_project_root():
    """Zwraca katalog główny repozytorium.

    Można go nadpisać zmienną środowiskową DEEPFAKE_PROJECT_ROOT, np. gdy
    dane leżą na innym dysku.
    """
    env_root = os.environ.get("DEEPFAKE_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parent.parent

# Ścieżka do aktualnego datasetu
def get_dataset_path():
    dataset_path = find_project_root() / "AudioFiles" / "Datasets" / "V2" / "compressed"
    return dataset_path

def get_dataset_split_csv(dataset_version):
    # 1 - V2 losowy (stary, z leakiem mic1/mic2)
    # 2 - V2 by_utterance (mic1+mic2 tej samej wypowiedzi razem)
    # 3 - V2 by_speaker (mówcy rozdzieleni między splity)
    # 4 - V2_mk (V2_small + OOD: CodecFake/DFADD/PeoplesSpeech w train)
    # 5 - V2 balans źródeł bonafide PO CZASIE + test = OOD (In-the-Wild)
    paths = {
        1: ("V2", "dataset_split.csv"),
        2: ("V2", "dataset_split_by_utterance.csv"),
        3: ("V2", "dataset_split_by_speaker.csv"),
        4: ("V2_mk", "dataset_split.csv"),
        5: ("V2", "dataset_split_balanced_ood.csv"),
    }
    if dataset_version not in paths:
        raise ValueError(f"Nieznana wersja datasetu: {dataset_version}. Dozwolone: {list(paths)}")
    folder, fname = paths[dataset_version]
    return find_project_root() / "AudioFiles" / "Datasets" / folder / fname


# Znajdź folder z plikami audio
def get_audio_folder():
    project_root = find_project_root()

    audio_folder = project_root.joinpath("AudioFiles")

    return audio_folder