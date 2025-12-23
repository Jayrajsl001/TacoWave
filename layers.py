# import torch
# from librosa.filters import mel as librosa_mel_fn
# from audio_processing import dynamic_range_compression
# from audio_processing import dynamic_range_decompression
# from stft import STFT


# class LinearNorm(torch.nn.Module):
#     def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
#         super(LinearNorm, self).__init__()
#         self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

#         torch.nn.init.xavier_uniform_(
#             self.linear_layer.weight,
#             gain=torch.nn.init.calculate_gain(w_init_gain))

#     def forward(self, x):
#         return self.linear_layer(x)


# class ConvNorm(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
#                  padding=None, dilation=1, bias=True, w_init_gain='linear'):
#         super(ConvNorm, self).__init__()
#         if padding is None:
#             assert(kernel_size % 2 == 1)
#             padding = int(dilation * (kernel_size - 1) / 2)

#         self.conv = torch.nn.Conv1d(in_channels, out_channels,
#                                     kernel_size=kernel_size, stride=stride,
#                                     padding=padding, dilation=dilation,
#                                     bias=bias)

#         torch.nn.init.xavier_uniform_(
#             self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

#     def forward(self, signal):
#         conv_signal = self.conv(signal)
#         return conv_signal


# class TacotronSTFT(torch.nn.Module):
#     def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
#                  n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
#                  mel_fmax=8000.0):
#         super(TacotronSTFT, self).__init__()
#         self.n_mel_channels = n_mel_channels
#         self.sampling_rate = sampling_rate
#         self.stft_fn = STFT(filter_length, hop_length, win_length)
#         mel_basis = librosa_mel_fn(
#             sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
#         mel_basis = torch.from_numpy(mel_basis).float()
#         self.register_buffer('mel_basis', mel_basis)

#     def spectral_normalize(self, magnitudes):
#         output = dynamic_range_compression(magnitudes)
#         return output

#     def spectral_de_normalize(self, magnitudes):
#         output = dynamic_range_decompression(magnitudes)
#         return output

#     def mel_spectrogram(self, y):
#         """Computes mel-spectrograms from a batch of waves
#         PARAMS
#         ------
#         y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

#         RETURNS
#         -------
#         mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
#         """
#         assert(torch.min(y.data) >= -1)
#         assert(torch.max(y.data) <= 1)

#         magnitudes, phases = self.stft_fn.transform(y)
#         magnitudes = magnitudes.data
#         mel_output = torch.matmul(self.mel_basis, magnitudes)
#         mel_output = self.spectral_normalize(mel_output)
#         return mel_output
# layers.py: Utility Layers for TacoWave
# Standard Tacotron2-derived layers with Xavier init and Conv1d support
# Compatible with Python 3.7 and V2 hparams (e.g., filter_length=2048, mel_fmax=11000)
# Added: to_gpu and get_mask_from_lengths to fix ImportError
# Fixed: get_mask_from_lengths returns bool mask for ~ operator compatibility

import torch
from librosa.filters import mel as librosa_mel_fn
from audio_processing import dynamic_range_compression
from audio_processing import dynamic_range_decompression
from stft import STFT


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len) if torch.cuda.is_available() else torch.LongTensor(max_len))
    mask = (lengths.unsqueeze(1) > ids)  # Bool mask for ~ compatibility
    return mask