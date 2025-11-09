import numpy as np
import matplotlib.pyplot as plt
import audioflux as af
from audioflux.display import fill_spec

# --- audio source ---
try:
    sample_path = af.utils.sample_path('220')
    audio_arr, sr = af.read(sample_path)
    if audio_arr.size == 0:
        raise RuntimeError("Empty sample")
except Exception:
    # fallback: synthesize 2 s of 220 Hz + harmonics
    sr = 32000
    t = np.linspace(0, 2.0, int(sr*2.0), endpoint=False)
    audio_arr = (np.sin(2*np.pi*220*t) +
                 0.5*np.sin(2*np.pi*440*t) +
                 0.25*np.sin(2*np.pi*660*t)).astype(np.float32)

# --- MEL spectrogram ---
spec_arr, mel_fre_band_arr = af.mel_spectrogram(
    audio_arr, num=128, radix2_exp=12, samplate=sr
)
spec_arr = np.abs(spec_arr) + 1e-12  # avoid log(0)

# --- MFCC ---
mfcc_arr, _ = af.mfcc(
    audio_arr, cc_num=13, mel_num=128, radix2_exp=12, samplate=sr
)

# --- Display ---
x_coords = np.linspace(0, audio_arr.shape[-1]/sr, spec_arr.shape[-1]+1)
y_coords = np.insert(mel_fre_band_arr, 0, 0)

fig, ax = plt.subplots()
img = fill_spec(20*np.log10(spec_arr), axes=ax,
                x_coords=x_coords, y_coords=y_coords,
                x_axis='time', y_axis='log',
                title='Mel Spectrogram (dB)')
fig.colorbar(img, ax=ax)

fig, ax = plt.subplots()
img = fill_spec(mfcc_arr, axes=ax,
                x_coords=x_coords, x_axis='time',
                title='MFCC (13 coeffs)')
fig.colorbar(img, ax=ax)

plt.show()
