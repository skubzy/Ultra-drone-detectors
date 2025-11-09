import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from collections import deque
import audioflux as af

# -------- SETTINGS --------
SR = 32000
WIN = 2**12       # 4096
HOP = 2**10       # 1024
MELS = 96
LOWF, HIGHF = 50, 8000
HISTORY_SEC = 8.0
CHANNELS = 1
VMIN, VMAX = -90, 0   # color scale in dB
DEVICE = None         # set to an input index if needed, e.g., 1
# --------------------------

max_samples = int(HISTORY_SEC * SR)
ring = deque(maxlen=max_samples)

def compute_mel_db(y):
    spec, mel_fb = af.mel_spectrogram(
        y, num=MELS, radix2_exp=int(np.log2(WIN)),
        samplate=SR, low_fre=LOWF, high_fre=HIGHF
    )
    spec_db = 20*np.log10(np.abs(spec) + 1e-12)
    return spec_db, mel_fb

def audio_callback(indata, frames, time, status):
    if status:
        print("Audio status:", status)
    x = indata[:, 0] if indata.ndim > 1 else indata
    ring.extend(x.tolist())

def main():
    # 0) Device check
    if DEVICE is not None:
        sd.default.device = DEVICE
    try:
        print("Available devices:")
        print(sd.query_devices())
    except Exception as e:
        print("Device query error:", e)

    # 1) Start stream
    with sd.InputStream(channels=CHANNELS, samplerate=SR,
                        blocksize=HOP, dtype='float32',
                        callback=audio_callback):
        print("Listening… speak or make a sound (fan works well).")
        fig, ax = plt.subplots(figsize=(10, 5))
        img = None
        cbar = None

        # UI loop
        while plt.fignum_exists(fig.number):
            n = len(ring)
            if n >= WIN:
                y = np.asarray(ring, dtype=np.float32)
                spec_db, mel_fb = compute_mel_db(y)
                dur = y.shape[0] / SR
                extent = [max(0, dur - HISTORY_SEC), dur, mel_fb[0], mel_fb[-1]]

                if img is None:
                    img = ax.imshow(spec_db, origin='lower', aspect='auto',
                                    extent=extent, vmin=VMIN, vmax=VMAX)
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Frequency (Hz, mel band edges)")
                    ax.set_title("Live Mel Spectrogram (dB)")
                    cbar = fig.colorbar(img, ax=ax)
                else:
                    img.set_data(spec_db)
                    img.set_extent(extent)

                plt.pause(0.08)  # ~12.5 FPS
            else:
                print(f"Buffering… {n}/{WIN} samples")
                plt.pause(0.2)

        print("Window closed, exiting.")

if __name__ == "__main__":
    main()
