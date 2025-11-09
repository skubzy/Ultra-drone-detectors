# live_mel_spectrogram.py  (no WindowType, no BFT)
import numpy as np
import sounddevice as sd; print(sd.query_devices()); sd.default.device = 1
import matplotlib.pyplot as plt
from collections import deque
import audioflux as af
from audioflux.display import fill_spec

# -------- SETTINGS --------
SR = 32000
WIN = 2**12          # FFT size implied by radix2_exp in mel_spectrogram
HOP = 2**10
MELS = 128
LOWF, HIGHF = 50, 8000
HISTORY_SEC = 8.0
REFRESH_HZ = 12
CHANNELS = 1
# --------------------------

max_samples = int(HISTORY_SEC * SR)
ring = deque(maxlen=max_samples)

def compute_mel_db(y):
    spec, mel_fb = af.mel_spectrogram(
        y, num=MELS, radix2_exp=int(np.log2(WIN)), samplate=SR,
        low_fre=LOWF, high_fre=HIGHF
    )
    return 20*np.log10(np.abs(spec) + 1e-12), mel_fb

fig, ax = plt.subplots(figsize=(10,5))
img = None
cbar = None

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    x = indata[:, 0] if indata.ndim > 1 else indata
    ring.extend(x.tolist())

def updater(_):
    global img, cbar
    if len(ring) < WIN:
        return img
    y = np.asarray(ring, dtype=np.float32)
    spec_db, mel_fb = compute_mel_db(y)
    duration = y.shape[0] / SR
    x_coords = np.linspace(0, duration, spec_db.shape[-1] + 1)
    y_coords = np.insert(mel_fb, 0, 0)

    ax.clear()
    img = fill_spec(spec_db, axes=ax, x_coords=x_coords, y_coords=y_coords,
                    x_axis='time', y_axis='log', title='Live Mel Spectrogram (dB)')
    if cbar is None:
        cbar = plt.colorbar(img, ax=ax)
    ax.set_xlim(max(0, duration - HISTORY_SEC), duration)
    plt.tight_layout()
    return img

def main():
    # print(sd.query_devices()); sd.default.device = <index>  # if needed
    with sd.InputStream(channels=CHANNELS, samplerate=SR, blocksize=HOP, callback=audio_callback):
        timer = fig.canvas.new_timer(interval=int(1000/REFRESH_HZ))
        timer.add_callback(updater, None)
        timer.start()
        print("Listening… close the window to stop.")
        plt.show()

if __name__ == "__main__":
    main()
