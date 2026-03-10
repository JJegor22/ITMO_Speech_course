from typing import Optional

import torch
from torch import nn
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio import functional as F

class LogMelFilterBanks(nn.Module):
    def __init__(
            self,
            n_fft: int = 400,
            samplerate: int = 16000,
            hop_length: int = 160,
            n_mels: int = 80,
            pad_mode: str = 'reflect',
            power: float = 2.0,
            normalize_stft: bool = False,
            onesided: bool = True,
            center: bool = True,
            return_complex: bool = True,
            f_min_hz: float = 0.0,
            f_max_hz: Optional[float] = None,
            norm_mel: Optional[str] = None,
            mel_scale: str = 'htk'
        ):
        super(LogMelFilterBanks, self).__init__()
        # general params and params defined by the exercise
        self.n_fft = n_fft
        self.window_length = n_fft
        self.window = torch.hann_window(self.window_length)
        self.power = power
        self.return_complex = return_complex

        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz if f_max_hz is not None else float(samplerate // 2)
        self.n_mels = n_mels
        self.samplerate = samplerate
        self.norm_mel = norm_mel
        self.mel_scale = mel_scale


        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.normalize_stft = normalize_stft
        self.onesided = onesided

        self.mel_fbanks = self._init_melscale_fbanks()

    def _init_melscale_fbanks(self):
        return F.melscale_fbanks(
            # Turns a normal STFT into a mel frequency STFT with triangular filter banks
            n_freqs=(self.n_fft // 2) + 1,
            f_min=self.f_min_hz,
            f_max=self.f_max_hz,
            n_mels=self.n_mels,
            sample_rate=self.samplerate,
            norm=self.norm_mel,
            mel_scale = self.mel_scale
        )

    def spectrogram(self, x):
        # x - is an input signal
        return torch.stft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalize_stft,
            onesided=self.onesided,
            return_complex=self.return_complex
        )

    def forward(self, x):
        """
        Args:
            x (Torch.Tensor): Tensor of audio of dimension (batch, time), audiosignal
        Returns:
            Torch.Tensor: Tensor of log mel filterbanks of dimension (batch, n_mels, n_frames),
                where n_frames is a function of the window_length, hop_length and length of audio
        """
        spectr = self.spectrogram(x)

        power_spectr = spectr.abs().pow(self.power)
        
        mel_spec = torch.matmul(power_spectr.transpose(-1, -2), self.mel_fbanks).transpose(-1, -2)
        
        log_mel_spec = torch.log(mel_spec + 1e-6)
        
        return log_mel_spec

to_download = False
data_path = ''
file_path = 'SpeechCommands/speech_commands_v0.02/forward/017c4098_nohash_0.wav'
data = SPEECHCOMMANDS(root=data_path, download=to_download)
signal, sr = torchaudio.load(data_path+file_path)
melspec = torchaudio.transforms.MelSpectrogram(
    hop_length=160,
    n_mels=80
)(signal)
logmelbanks = LogMelFilterBanks()(signal)

assert torch.log(melspec + 1e-6).shape == logmelbanks.shape
assert torch.allclose(torch.log(melspec + 1e-6), logmelbanks)