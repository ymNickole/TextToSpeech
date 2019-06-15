import wave

import librosa
import librosa.filters
import math
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
from cn.hparams import hparams


def load_wav(path):
    return librosa.core.load(path, sr=hparams.sample_rate)[0]


def save_wav(wav, path):
    # rescaling for unified measure for all clips
    wav = wav / np.abs(wav).max() * 0.999
    # factor 0.5 in case of overflow for int16
    f1 = 0.5 * 32767 / max(0.01, np.max(np.abs(wav)))
    # sublinear scaling as Y ~ X ^ k (k < 1)
    f2 = np.sign(wav) * np.power(np.abs(wav), 0.8)
    wav = f1 * f2
    # bandpass for less noises
    firwin = signal.firwin(hparams.num_freq, [hparams.fmin, hparams.fmax], pass_zero=False, fs=hparams.sample_rate)
    wav = signal.convolve(wav, firwin)

    wavfile.write(path, hparams.sample_rate, wav.astype(np.int16))


def trim_silence(wav):
    return librosa.effects.trim(wav, top_db=60, frame_length=512, hop_length=128)[0]

# 保证调用时wav_files列表非仅有1个wav
def merge_wavs_with_standstill(wav_files, standstill_time, output_filename):
    with wave.open(wav_files[0].name, 'rb') as first_wav:
        params = first_wav.getparams()
        framerate = first_wav.getframerate()
        first_nframes = first_wav.getnframes()
        first_wav_str = first_wav.readframes(first_nframes)
    first_wav_data = np.fromstring(first_wav_str, dtype=np.int16)
    sum_frames = first_nframes
    result = first_wav_data

    for i in range(1, len(wav_files)):
        nframes_of_standstill = int(framerate * standstill_time / 1000)
        sum_frames += nframes_of_standstill
        standstill_data = np.zeros(nframes_of_standstill, dtype=np.int16)
        result = np.concatenate((result, standstill_data))

        with wave.open(wav_files[i].name, 'rb') as wav:
            nframes = wav.getnframes()
            sum_frames += nframes
            wav_str = wav.readframes(nframes)
        wav_data = np.fromstring(wav_str, dtype=np.int16)
        result = np.concatenate((result, wav_data))

    with wave.open(output_filename, 'wb') as output:
        output.setparams(params[:6])
        output.setnframes(sum_frames)
        output.writeframes(result.tostring())  # outData:16位，-32767~32767，注意不要溢出
        return output


def preemphasis(x):
    return signal.lfilter([1, -hparams.preemphasis], [1], x)


def inv_preemphasis(x):
    return signal.lfilter([1], [1, -hparams.preemphasis], x)


def spectrogram(y):
    D = _stft(preemphasis(y))
    S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
    return _normalize(S)


def inv_spectrogram(spectrogram):
    '''Converts spectrogram to waveform using librosa'''
    S = _db_to_amp(_denormalize(spectrogram) + hparams.ref_level_db)  # Convert back to linear
    return inv_preemphasis(_griffin_lim(S ** hparams.power))  # Reconstruct phase


def inv_spectrogram_tensorflow(spectrogram):
    '''Builds computational graph to convert spectrogram to waveform using TensorFlow.

  Unlike inv_spectrogram, this does NOT invert the preemphasis. The caller should call
  inv_preemphasis on the output after running the graph.
  '''
    S = _db_to_amp_tensorflow(_denormalize_tensorflow(spectrogram) + hparams.ref_level_db)
    return _griffin_lim_tensorflow(tf.pow(S, hparams.power))


def melspectrogram(y):
    D = _stft(preemphasis(y))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
    return _normalize(S)


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
    window_length = int(hparams.sample_rate * min_silence_sec)
    hop_length = int(window_length / 4)
    threshold = _db_to_amp(threshold_db)
    for x in range(hop_length, len(wav) - window_length, hop_length):
        if np.max(wav[x:x + window_length]) < threshold:
            return x + hop_length
    return len(wav)


def _griffin_lim(S):
    '''librosa implementation of Griffin-Lim
  Based on https://github.com/librosa/librosa/issues/434
  '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y


def _griffin_lim_tensorflow(S):
    '''TensorFlow implementation of Griffin-Lim
  Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
  '''
    with tf.variable_scope('griffinlim'):
        # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
        S = tf.expand_dims(S, 0)
        S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
        y = _istft_tensorflow(S_complex)
        for i in range(hparams.griffin_lim_iters):
            est = _stft_tensorflow(y)
            angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
            y = _istft_tensorflow(S_complex * angles)
        return tf.squeeze(y, 0)


def _stft(y):
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):
    _, hop_length, win_length = _stft_parameters()
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_tensorflow(signals):
    n_fft, hop_length, win_length = _stft_parameters()
    return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)


def _istft_tensorflow(stfts):
    n_fft, hop_length, win_length = _stft_parameters()
    return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)


def _stft_parameters():
    n_fft = (hparams.num_freq - 1) * 2
    hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
    return n_fft, hop_length, win_length


# Conversions:

_mel_basis = None


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis():
    n_fft = (hparams.num_freq - 1) * 2
    assert hparams.fmax < hparams.sample_rate // 2
    return librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=hparams.num_mels, fmin=hparams.fmin,
                               fmax=hparams.fmax)


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _db_to_amp_tensorflow(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)


def _normalize(S):
    # symmetric mels
    return 2 * hparams.max_abs_value * ((S - hparams.min_level_db) / -hparams.min_level_db) - hparams.max_abs_value


def _denormalize(S):
    # symmetric mels
    return ((S + hparams.max_abs_value) * -hparams.min_level_db) / (2 * hparams.max_abs_value) + hparams.min_level_db


def _denormalize_tensorflow(S):
    # symmetric mels
    return ((S + hparams.max_abs_value) * -hparams.min_level_db) / (2 * hparams.max_abs_value) + hparams.min_level_db
