import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import audioflux as af
from audioflux.display import fill_spec

def main():
    p = argparse.ArgumentParser(description="Make mel-spectrogram and MFCC images from a WAV.")
    p.add_argument("wav", help="Path to input WAV file")
    p.add_argument("--mels", type=int, default=128, help="Number of mel bins (default: 128)")
    p.add_argument("--fmin", type=float, default=50.0, help="Min freq for mel (Hz)")
    p.add_argument("--fmax", type=float, default=8000.0, help="Max freq for mel (Hz)")
    p.add_argument("--radix2", type=int, default=12, help="FFT size as 2**radix2 (default: 12 -> 4096)")
    args = p.parse_args()

    wav_path = Path(args.wav)
    out_mel = wav_path.with_suffix(".mel.png")
    out_mfcc = wav_path.with_suffix(".mfcc.png")

    # 1) Read audio
    y, sr = af.read(str(wav_path))
    if y.ndim > 1:
        y = y.mean(axis=-1)  # mono
    dur = y.shape[-1] / sr

    # 2) Mel spectrogram -> dB
    spec, mel_fb = af.mel_spectrogram(
        y, num=args.mels, radix2_exp=args.radix2,
        samplate=sr, low_fre=args.fmin, high_fre=args.fmax
    )
    spec_db = 20*np.log10(np.abs(spec) + 1e-12)

    # time/freq coords for plotting
    x = np.linspace(0, dur, spec_db.shape[-1] + 1)
    y_coords = np.insert(mel_fb, 0, 0)

    # Plot mel-spectrogram
    fig, ax = plt.subplots(figsize=(10, 5))
    img = fill_spec(spec_db, axes=ax, x_coords=x, y_coords=y_coords,
                    x_axis='time', y_axis='log', title='Mel Spectrogram (dB)')
    fig.colorbar(img, ax=ax)
    plt.tight_layout()
    fig.savefig(out_mel, dpi=200)
    plt.close(fig)

    # 3) MFCCs
    mfcc, _ = af.mfcc(y, cc_num=13, mel_num=args.mels,
                      radix2_exp=args.radix2, samplate=sr,
                      low_fre=args.fmin, high_fre=args.fmax)

    x_m = np.linspace(0, dur, mfcc.shape[-1] + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = fill_spec(mfcc, axes=ax, x_coords=x_m, x_axis='time',
                    title='MFCC (13 coefficients)')
    fig.colorbar(img, ax=ax)
    plt.tight_layout()
    fig.savefig(out_mfcc, dpi=200)
    plt.close(fig)

    print(f"Saved:\n  {out_mel}\n  {out_mfcc}")

if __name__ == "__main__":
    main()
