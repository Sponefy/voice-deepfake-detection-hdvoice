from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def create_dataset_csv(
    source_dir: str,
    output_csv: str = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Tworzy CSV z podziałem datasetu na train/val/test.

    Returns:
        DataFrame z kolumnami: file_path, label, split
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    source_path = Path(source_dir)

    if output_csv is None:
        output_csv = source_path.parent / "dataset_split.csv"
    else:
        output_csv = Path(output_csv)

    audio_extensions = {'.wav', '.mp3', '.aac', '.m4a', '.flac', '.ogg'}

    data = []
    categories = ['bonafide', 'deepfake']

    for category in categories:
        category_path = source_path / category

        if not category_path.exists():
            continue

        files = [f for f in category_path.rglob('*')
                if f.suffix.lower() in audio_extensions]

        train_files, temp_files = train_test_split(
            files,
            test_size=(val_ratio + test_ratio),
            random_state=random_state
        )

        val_files, test_files = train_test_split(
            temp_files,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=random_state
        )

        for file in train_files:
            data.append({'file_path': str(file), 'label': category, 'split': 'train'})
        for file in val_files:
            data.append({'file_path': str(file), 'label': category, 'split': 'val'})
        for file in test_files:
            data.append({'file_path': str(file), 'label': category, 'split': 'test'})

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"✓ Zapisano CSV: {output_csv}")
    print(f"\nPodział:")
    print(df.groupby(['split', 'label']).size().unstack(fill_value=0))

    return df


class AudioDatasetFromCSV(Dataset):
    """
    Dataset audio z CSV-em podziału train/val/test.

    Parametry:
        csv_file: ścieżka do CSV (kolumny: file_path, label, split)
        split:    'train' | 'val' | 'test'
        transform: opcjonalna funkcja na audio (np.ndarray) po cropping/padding
        max_length: długość docelowa w próbkach (16 kHz → 64000 = 4 s)
        random_crop: True dla treningu — losowy wycinek dla długich plików.
                     False dla val/test — środkowy crop (deterministyczny).
                     Padding (dla krótkich nagrań) zawsze z prawej strony,
                     żeby zachować konwencję HuggingFace Wav2Vec2 dla
                     attention_mask.

    Zwraca:
        input_values:   FloatTensor [max_length] — audio (z paddingiem zerami)
        attention_mask: LongTensor  [max_length] — 1 dla realnego sygnału,
                                                   0 dla paddingu
        labels:         LongTensor  [1]
    """

    def __init__(
        self,
        csv_file: str,
        split: str = 'train',
        transform=None,
        max_length: int = 64000,
        random_crop: bool | None = None,
    ):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.transform = transform
        self.max_length = max_length
        # Domyślnie: random crop tylko na treningu
        self.random_crop = (split == 'train') if random_crop is None else random_crop

    def __len__(self):
        return len(self.df)

    @property
    def labels_array(self) -> np.ndarray:
        """np.ndarray etykiet 0/1 (deepfake/bonafide) bez wczytywania audio.
        Wykorzystywane do budowy WeightedRandomSampler."""
        return (self.df['label'].values == 'bonafide').astype(np.int64)

    def __getitem__(self, idx):
        import soundfile as sf

        row = self.df.iloc[idx]
        audio_path = row['file_path']
        label = 1 if row['label'] == 'bonafide' else 0  # bonafide=1, deepfake=0

        audio, sr = sf.read(audio_path)

        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        target = self.max_length
        n = len(audio)

        if n > target:
            # Długie nagranie — losowy (trening) lub środkowy (val/test) wycinek
            if self.random_crop:
                start = np.random.randint(0, n - target + 1)
            else:
                start = (n - target) // 2
            audio = audio[start:start + target]
            real_len = target
        elif n < target:
            # Krótkie nagranie — padding zerami z prawej strony.
            # Pozycja nie jest randomizowana, bo attention_mask zajmuje się
            # tym, żeby model i tak nie korzystał z paddingu.
            audio = np.pad(audio, (0, target - n))
            real_len = n
        else:
            real_len = target

        attention_mask = np.zeros(target, dtype=np.int64)
        attention_mask[:real_len] = 1

        if self.transform:
            audio = self.transform(audio)

        return {
            'input_values': torch.FloatTensor(audio),
            'attention_mask': torch.LongTensor(attention_mask),
            'labels': torch.LongTensor([label]),
        }


def build_class_balanced_sampler(dataset: 'AudioDatasetFromCSV'):
    """
    Tworzy WeightedRandomSampler dający każdej klasie równe prawdopodobieństwo
    pojawienia się w batchu, niezależnie od tego ile plików jest w klasie.

    Każdy element dostaje wagę 1/N_klasa. Suma wag w każdej klasie wynosi 1,
    więc szansa wylosowania bonafide = szansa wylosowania deepfake.

    num_samples = len(dataset), replacement=True — jeden epoch ma tyle samo
    iteracji co dotychczas, ale klasa mniejszościowa jest oversampled.
    """
    from torch.utils.data import WeightedRandomSampler

    labels = dataset.labels_array
    counts = np.bincount(labels, minlength=2)
    if (counts == 0).any():
        raise ValueError(f"Brak jednej z klas w datasecie: counts={counts}")

    per_class_weight = 1.0 / counts                        # [w_deepfake, w_bonafide]
    sample_weights = per_class_weight[labels]              # waga dla każdego pliku

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(labels),
        replacement=True,
    )