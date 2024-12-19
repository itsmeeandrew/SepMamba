import torchaudio.transforms as T
import torch
import random

class Preprocessor:
    def __init__(self, resample_rate, to_length=None):
        self.resample_rate = resample_rate
        self.to_length = to_length

    def __call__(self, waveform, sample_rate):
        waveform = self.resample(waveform, sample_rate)
        if self.to_length != 0:
            waveform = self.cut_to_length(waveform)
        waveform = self.clear_nans(waveform)

        return waveform

    def _get_resampler(self, waveform, sample_rate):
        return T.Resample(sample_rate, self.resample_rate, dtype=waveform.dtype)

    def resample(self, waveform, sample_rate):
        resampler = self._get_resampler(waveform, sample_rate)

        return resampler(waveform)

    def clear_nans(self, waveform):
        waveform = torch.where(waveform == 0, random.uniform(-1e-9, 1e-9), waveform)
        return waveform
    
    def cut_to_length(self, waveform):
        audio_length = waveform.shape[1]

        if audio_length > self.to_length:
            start_idx = random.randint(0, audio_length - self.to_length - 1)

            waveform = waveform[:, start_idx:(start_idx+self.to_length)]
        else:
            zero_pad = torch.full([1, self.to_length - audio_length], 0)

            waveform = torch.cat((waveform, zero_pad), dim=1)

        return waveform