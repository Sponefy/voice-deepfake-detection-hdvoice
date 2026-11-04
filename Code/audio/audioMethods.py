from pydub import AudioSegment
from pathlib import Path
from pydub.effects import compress_dynamic_range
import torch
import torchaudio
import torchaudio.transforms as T
from torchaudio.functional import highpass_biquad, lowpass_biquad   # Explicit import
import subprocess
import tempfile
import random


def load_audio_auto_format(audio_path):
    """Automatycznie wykrywa format audio na podstawie rozszerzenia lub pozwala pydub go wykryć"""
    audio_path = Path(audio_path)

    extension = audio_path.suffix.lower()
    format_map = {
        '.wav': 'wav',
        '.mp3': 'mp3',
        '.aac': 'aac',
        '.m4a': 'm4a',
        '.flac': 'flac',
        '.ogg': 'ogg'
    }

    if extension in format_map:
        return AudioSegment.from_file(audio_path, format=format_map[extension])
    else:
        return AudioSegment.from_file(audio_path)

# nie używać tego, bo jest bardzo wolne
def compress_audio(input_path, output_path, format="wav"):
    """Podstawowa kompresja HD Voice (bez symulacji kodeka)"""
    sample_rate = 16000
    high_pass = 50
    low_pass = 7000

    audio = AudioSegment.from_file(input_path)

    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.high_pass_filter(high_pass)
    audio = audio.low_pass_filter(low_pass)
    audio = audio.normalize()
    audio = compress_dynamic_range(audio, threshold=-20.0, ratio=4.0, attack=5.0)

    audio.export(output_path, format=format)

_choices = [
    (12.65, 0.55),
    (8.85, 0.15),
    (23.85, 0.15),
    (6.60, 0.025),
    (14.25, 0.025),
    (15.85, 0.025),
    (18.25, 0.025),
    (19.85, 0.025),
    (23.05, 0.025),
]

def pick_mode() -> str:
    """Return a bitrate string suitable for ffmpeg (e.g. '12.65k') using the weighted choices."""
    r = random.random()
    acc = 0.0
    for bitrate, p in _choices:
        acc += p
        if r <= acc:
            return f"{bitrate}k"
    # fallback (shouldn't happen if probabilities sum to 1)
    return f"{_choices[0][0]}k"


# szybka symulacja HD Voice używając torchaudio + FFmpeg
# ~10x szybsza niż pydub dla dużych zbiorów danych
def compress_audio_hd_voice_fast(input_path: str, output_path: str) -> None:

    # Stwórz katalog wyjściowy jeśli nie istnieje
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 1. Załaduj audio
    waveform, sr = torchaudio.load(input_path)

    # 2. Konwersja do mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 3. Resample do 16 kHz
    if sr != 16000:
        resampler = T.Resample(sr, 16000)
        waveform = resampler(waveform)
        sr = 16000

    # 4. Bandpass filter (50-7000 Hz)
    waveform = apply_bandpass_filter(waveform, sr, low_freq=50, high_freq=7000)

    # 5. Normalizacja
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val

    # 6. Symulacja kodeka przez FFmpeg
    with tempfile.NamedTemporaryFile(suffix='.opus', delete=False) as temp_opus:
        temp_opus_path: str = temp_opus.name

    try:
        # Zapisz do tymczasowego WAV
        temp_wav: str = temp_opus_path.replace('.opus', '_input.wav')
        torchaudio.save(temp_wav, waveform, sr)

        chosen_bitrate = pick_mode()

        # Kompresja Opus (tryb VoIP) z losowym bitratem z trybów AMR-WB (6.6–23.85 kbps)
        result = subprocess.run([
            'ffmpeg', '-y', '-i', temp_wav,
            '-c:a', 'libopus',
            '-b:a', chosen_bitrate,
            '-application', 'voip',
            temp_opus_path
        ], capture_output=True, text=True, encoding='utf-8', errors='replace')

        if result.returncode != 0:
            print(f"FFmpeg encode error: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)

        # Dekompresja z powrotem do WAV (bezpośrednio przez torchaudio, bez FFmpeg)
        temp_decoded_waveform, decoded_sr = torchaudio.load(temp_opus_path)

        # Konwersja do mono i 16kHz
        if temp_decoded_waveform.shape[0] > 1:
            temp_decoded_waveform = torch.mean(temp_decoded_waveform, dim=0, keepdim=True)

        # Resample tylko jeśli sample rate się różni
        if decoded_sr != 16000:
            resampler = T.Resample(decoded_sr, 16000)  # Użyj wykrytego SR zamiast 48000
            temp_decoded_waveform = resampler(temp_decoded_waveform)

        # Zapisz finalny plik
        torchaudio.save(output_path, temp_decoded_waveform, 16000)

    finally:
        # Cleanup
        Path(temp_opus_path).unlink(missing_ok=True)
        Path(temp_wav).unlink(missing_ok=True)


def apply_bandpass_filter(waveform: torch.Tensor, sr: int,
                          low_freq: int = 50, high_freq: int = 7000) -> torch.Tensor:

    # High-pass filter
    highpass = highpass_biquad(waveform, sr, cutoff_freq=low_freq)
    waveform = highpass

    # Low-pass filter
    lowpass = lowpass_biquad(waveform, sr, cutoff_freq=high_freq)
    waveform = lowpass

    return waveform