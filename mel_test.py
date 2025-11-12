import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import audioflux as af
from audioflux.display import fill_spec

# --------- paths ---------
sample_path = Path(r"C:\Users\Pc\Desktop\Desktop\Stuff\Ultra-drone-detectors\UAVDataset\fan\Noisy-Nodrone\rec_20251111T190323Z_000.wav")
output_dir  = Path(r"C:\Users\Pc\Desktop\Desktop\Stuff\Ultra-drone-detectors\data\non-drone")  # <-- change to your project folder
output_dir.mkdir(parents=True, exist_ok=True)

stem = sample_path.stem  # e.g., "briskaudioclip2"

# --------- read & mono ---------
audio_arr, sr = af.read(str(sample_path))
if audio_arr.ndim > 1:
    audio_arr = audio_arr.mean(axis=-1)

# --------- MEL (dB) ---------
spec_arr, mel_fre_band_arr = af.mel_spectrogram(
    audio_arr, num=128, radix2_exp=12, samplate=sr
)
spec_db = 20 * np.log10(np.abs(spec_arr) + 1e-12)

# --------- MFCC ---------
mfcc_arr, _ = af.mfcc(
    audio_arr, cc_num=13, mel_num=128, radix2_exp=12, samplate=sr
)
mfcc_arr = np.asarray(mfcc_arr, dtype=np.float32)

# --------- axes for plotting ---------
audio_len = audio_arr.shape[-1] / sr
x_coords = np.linspace(0, audio_len, spec_db.shape[-1] + 1)
y_coords = np.insert(mel_fre_band_arr, 0, 0)
x_m = np.linspace(0, audio_len, mfcc_arr.shape[-1] + 1)

# --------- save MEL figure ---------
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
img = fill_spec(spec_db, axes=ax, x_coords=x_coords, y_coords=y_coords,
                x_axis='time', y_axis='log', title='Mel Spectrogram (dB)')
fig.colorbar(img, ax=ax)
mel_png = output_dir / f"{stem}_mel.png"
fig.savefig(mel_png, bbox_inches='tight', pad_inches=0.05)
plt.close(fig)

# --------- save MFCC figure ---------


# --------- (optional) save arrays for ML ---------
np.save(output_dir / f"{stem}_mel_db.npy", spec_db)   # shape: (n_mels, T)
np.save(output_dir / f"{stem}_mfcc.npy", mfcc_arr)    # shape: (13, T)
np.save(output_dir / f"{stem}_audio.npy", audio_arr)  # raw mono waveform

print(f"Saved:\n- {mel_png}\n- {output_dir / (stem+'_mel_db.npy')}\n- {output_dir / (stem+'_mfcc.npy')}\n- {output_dir / (stem+'_audio.npy')}")
