import dataclasses
import os

import note_seq
import numpy
import torch
import tensorflow as tf

from constant import DEFAULT_SAMPLE_RATE, DEFAULT_HOP_WIDTH, DEFAULT_NUM_MEL_BINS, FFT_SIZE, MEL_LO_HZ


@dataclasses.dataclass
class SpectrogramConfig:
    sample_rate: int = DEFAULT_SAMPLE_RATE
    hop_width: int = DEFAULT_HOP_WIDTH
    num_mel_bins: int = DEFAULT_NUM_MEL_BINS

    @property
    def frames_per_second(self):
        return self.sample_rate / self.hop_width


def tf_float32(x):
    """Ensure array/tensor is a float32 tf.Tensor."""
    if isinstance(x, tf.Tensor):
        return tf.cast(x, dtype=tf.float32)  # This is a no-op if x is float32.
    else:
        return tf.convert_to_tensor(x, tf.float32)


def safe_log(x, eps=1e-5):
    """Avoid taking the log of a non-positive number."""
    safe_x = tf.where(x <= 0.0, eps, x)
    return tf.math.log(safe_x)


def stft(audio, frame_size=2048, overlap=0.75, pad_end=True):
    """Differentiable stft in tensorflow, computed in batch."""
    # Remove channel dim if present.
    audio = tf_float32(audio)
    if len(audio.shape) == 3:
        audio = tf.squeeze(audio, axis=-1)

    s = tf.signal.stft(
        signals=audio,
        frame_length=int(frame_size),
        frame_step=int(frame_size * (1.0 - overlap)),
        fft_length=None,  # Use enclosing power of 2.
        pad_end=pad_end)
    return s


def compute_mag(audio, size=2048, overlap=0.75, pad_end=True):
    mag = tf.abs(stft(audio, frame_size=size, overlap=overlap, pad_end=pad_end))
    return tf_float32(mag)


def compute_mel(audio,
                lo_hz=0.0,
                hi_hz=8000.0,
                bins=64,
                fft_size=2048,
                overlap=0.75,
                pad_end=True,
                sample_rate=16000):
    """Calculate Mel Spectrogram."""
    mag = compute_mag(audio, fft_size, overlap, pad_end)
    num_spectrogram_bins = int(mag.shape[-1])
    linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        bins, num_spectrogram_bins, sample_rate, lo_hz, hi_hz)
    mel = tf.tensordot(mag, linear_to_mel_matrix, 1)
    mel.set_shape(mag.shape[:-1].concatenate(linear_to_mel_matrix.shape[-1:]))
    return mel


def compute_logmel(audio,
                   lo_hz=80.0,
                   hi_hz=7600.0,
                   bins=64,
                   fft_size=2048,
                   overlap=0.75,
                   pad_end=True,
                   sample_rate=16000):
    """Logarithmic amplitude of mel-scaled spectrogram."""
    mel = compute_mel(audio, lo_hz, hi_hz, bins,
                      fft_size, overlap, pad_end, sample_rate)
    return safe_log(mel)

def split_audio(samples, spectrogram_config: SpectrogramConfig):
    x = torch.tensor(samples)
    padded = torch.nn.functional.pad(x, (0, spectrogram_config.hop_width - 1))
    frames = padded.unfold(0, spectrogram_config.hop_width, spectrogram_config.hop_width)
    return frames


def compute_spectrogram(samples, spectrogram_config):
    overlap = 1 - (spectrogram_config.hop_width / FFT_SIZE)
    return compute_logmel(
        samples,
        bins=spectrogram_config.num_mel_bins,
        lo_hz=MEL_LO_HZ,
        overlap=overlap,
        fft_size=FFT_SIZE,
        sample_rate=spectrogram_config.sample_rate).numpy()


def read_wave(path, sample_rate) -> numpy.ndarray:
    """

    Args:
        absolute_path: 路径
        sample_rate: 采样率

    Returns: 采样结果

    """
    if not os.path.exists(path):
        raise ValueError(f"未知文件:{path}")
    with open(path, "rb") as f:
        return note_seq.audio_io.wav_data_to_samples_librosa(f.read(), sample_rate=sample_rate)


def build_default_spectrogram_config():
    return SpectrogramConfig
