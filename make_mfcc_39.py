import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import audioflux as af
from audioflux.display import fill_spec

def compute_delta(feat, N=2):
    """
    HTK-style delta:
    Δ[t] = sum_{n=1..N} n*(c[t+n] - c[t-n]) / (2*sum_{n=1..N} n^2)
    feat shape: (C, T)
    """
    C, T = feat.shape
    denom = 2 * sum(n*n for n in range(1, N+1))
    padded = np.pad(feat, ((0,0),(N,N)), mode='edge')
    delta = np.zeros_like(feat, dtype=np.float32)
    for t in range(T):
        num = 0.0
        for n in range(1, N+1):
            num += n * (padded[:, t+N+n] - padded[:, t+N-n])
        delta[:,t] = num / denom
    return delta

def plot_feat(feat, title, out_png, time_sec):
    # feat: (C, T)
    x = np.linspace(0, time_sec, feat.shape[1]+1)
    fig, ax = plt.subplots(figsize=(12, 4))
    img = fill_spec(feat, axes=ax, x_coords=x, x_axis='time', title=title)
    fig.colorbar(img, ax=ax)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def main():
    p = argparse.ArgumentParser(description="Make MFCC + Δ + ΔΔ (39-D) and plots.")
    p.add_argument("wav", help="Path to WAV")
    p.add_argument("--mels", type=int, default=128)
    p.add_argument("--cc", type=int, default=13, help="Number of MFCCs (incl. c0)")
    p.add_argument("--radix2", type=int, default=12, help="FFT size 2**radix2")
    p.add_argument("--fmin", type=float, default=50.0)
    p.add_argument("--fmax", type=float, default=8000.0)
    p.add_argument("--drop_c0", action="store_true", help="Drop c0 for level invariance")
    args = p.parse_args()

    wav = Path(args.wav)
    out_base = wav.with_suffix("")

    # 1) Read audio
    y, sr = af.read(str(wav))
    if y.ndim > 1:
        y = y.mean(axis=-1)
    dur = y.shape[-1] / sr

    # 2) MFCCs (shape: (cc, T))
    mfcc, _ = af.mfcc(
        y, cc_num=args.cc, mel_num=args.mels,
        radix2_exp=args.radix2, samplate=sr,
        low_fre=args.fmin, high_fre=args.fmax
    )
    mfcc = np.asarray(mfcc, dtype=np.float32)  # <-- cast the ARRAY, not the tuple


    # Optionally drop c0
    if args.drop_c0 and mfcc.shape[0] > 12:
        mfcc = mfcc[1:, :]  # keep c1..c12

    # 3) Δ and ΔΔ
    delta = compute_delta(mfcc, N=2)
    deltadelta = compute_delta(delta, N=2)

    # 4) Stack features (39-D if cc=13 and no drop)
    feat39 = np.vstack([mfcc, delta, deltadelta]).astype(np.float32)

    # 5) Save arrays for training
    np.save(out_base.with_suffix(".mfcc.npy"), mfcc)
    np.save(out_base.with_suffix(".delta.npy"), delta)
    np.save(out_base.with_suffix(".deltadelta.npy"), deltadelta)
    np.save(out_base.with_suffix(".mfcc39.npy"), feat39)

    # 6) Plots
    plot_feat(mfcc, "MFCC ({} coeffs)".format(mfcc.shape[0]), out_base.with_suffix(".mfcc.png"), dur)
    plot_feat(delta, "Δ MFCC", out_base.with_suffix(".delta.png"), dur)
    plot_feat(deltadelta, "ΔΔ MFCC", out_base.with_suffix(".deltadelta.png"), dur)

    print("Saved:")
    print("  ", out_base.with_suffix(".mfcc.npy"))
    print("  ", out_base.with_suffix(".delta.npy"))
    print("  ", out_base.with_suffix(".deltadelta.npy"))
    print("  ", out_base.with_suffix(".mfcc39.npy"))
    print("  ", out_base.with_suffix(".mfcc.png"))
    print("  ", out_base.with_suffix(".delta.png"))
    print("  ", out_base.with_suffix(".deltadelta.png"))

if __name__ == "__main__":
    main()
