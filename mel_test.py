import numpy as np
import matplotlib.pyplot as plt
import audioflux as af
from audioflux.display import fill_spec
from pathlib import Path

# ---- choose ONE of these two sources ----
# 1) Your file:
sample_path = str(Path(r"C:\Users\Pc\Desktop\Desktop\Stuff\test123\mic_test.wav"))
# 2) Or the built-in sample:
# sample_path = af.utils.sample_path('220')  # may be empty on some installs

# Read audio (mono)
audio_arr, sr = af.read(sample_path)
if audio_arr.ndim > 1:
    audio_arr = audio_arr.mean(axis=-1)

# Mel spectrogram -> dB
spec_arr, mel_fre_band_arr = af.mel_spectrogram(
    audio_arr, num=128, radix2_exp=12, samplate=sr
)
spec_db = 20*np.log10(np.abs(spec_arr) + 1e-12)

# MFCC (returns (mfcc, mel_band_edges))
mfcc_arr, _ = af.mfcc(
    audio_arr, cc_num=13, mel_num=128, radix2_exp=12, samplate=sr
)
mfcc_arr = np.asarray(mfcc_arr, dtype=np.float32)

# Time/freq axes
audio_len = audio_arr.shape[-1] / sr
x_coords = np.linspace(0, audio_len, spec_db.shape[-1] + 1)
y_coords = np.insert(mel_fre_band_arr, 0, 0)

# Plot mel
fig, ax = plt.subplots()
img = fill_spec(spec_db, axes=ax, x_coords=x_coords, y_coords=y_coords,
                x_axis='time', y_axis='log', title='Mel Spectrogram (dB)')
fig.colorbar(img, ax=ax)

# Plot MFCC
x_m = np.linspace(0, audio_len, mfcc_arr.shape[-1] + 1)
fig, ax = plt.subplots()
img = fill_spec(mfcc_arr, axes=ax, x_coords=x_m, x_axis='time', title='MFCC (13)')
fig.colorbar(img, ax=ax)

plt.show()
