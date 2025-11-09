import argparse
from pathlib import Path
import numpy as np
import audioflux as af
import soundfile as sf
import csv

# --------- defaults ----------
SR_TARGET   = 32000
N_MELS      = 128
LOW_F, HI_F = 50.0, 8000.0
RADIX2_EXP  = 12          # FFT size = 2**12 = 4096
HOP_S       = 0.010       # ~10 ms hop (frame rate ~100 Hz)
SMOOTH_MS   = 200         # f0 smoothing (moving avg)
BLOCK_S     = 1.0         # aggregate window for CSV rows
HARM_BW     = 0.08        # +-8% bandwidth around harmonic for TNR mask
MAX_HARM    = 8           # # of harmonics to consider
ENV_BAND    = (200, 3000) # band for envelope/modulation
MOD_LO_HZ   = 10.0
MOD_HI_HZ   = 80.0
# -----------------------------

def resample_if_needed(y, sr):
    if sr == SR_TARGET:
        return y, sr
    # use audioflux resampler for simplicity
    y_rs = af.resample(y, samplate_orig=sr, samplate_new=SR_TARGET)
    return y_rs.astype(np.float32), SR_TARGET

def moving_avg(x, win):
    if win <= 1: return x
    k = int(win)
    if k % 2 == 0: k += 1
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    c = np.ones(k, dtype=np.float32) / k
    return np.convolve(xpad, c, mode="valid")

def hz_to_mel_freq_centers(mel_edges):
    # audioflux returns band edges; centers are midpoints
    return 0.5 * (mel_edges[:-1] + mel_edges[1:])

def mel_band_index_for_hz(centers_hz, hz):
    # nearest mel bin index for a target frequency
    return int(np.clip(np.argmin(np.abs(centers_hz - hz)), 0, len(centers_hz) - 1))

def compute_logmel(y, sr):
    spec, mel_edges = af.mel_spectrogram(
        y, num=N_MELS, radix2_exp=RADIX2_EXP,
        samplate=sr, low_fre=LOW_F, high_fre=HI_F
    )
    mel = np.abs(spec).astype(np.float32) + 1e-12
    mel_db = 20.0 * np.log10(mel)
    return mel_db, mel_edges

def yin_f0(y, sr, hop_len):
    pitch = af.PitchYIN(samplate=sr, frame_length=2**RADIX2_EXP, hop_length=hop_len)
    f0 = pitch.pitch(y).astype(np.float32)  # Hz
    f0[f0 < 1.0] = 0.0
    return f0

def spectral_flux(mel):
    # mel: (n_mels, T); flux over time (T,)
    diff = np.diff(mel, axis=1)
    diff[diff < 0] = 0
    flux = diff.sum(axis=0)
    flux = np.concatenate([[flux[0]], flux])  # same length as T
    return flux.astype(np.float32)

def band_envelope(y, sr, lo, hi):
    # crude bandpass: subtract mel bands outside range and collapse to RMS envelope
    spec, edges = af.mel_spectrogram(y, num=N_MELS, radix2_exp=RADIX2_EXP,
                                     samplate=sr, low_fre=LOW_F, high_fre=HI_F)
    mel = np.abs(spec).astype(np.float32) + 1e-12
    centers = hz_to_mel_freq_centers(edges)
    keep = (centers >= lo) & (centers <= hi)
    env = np.sqrt((mel[keep]**2).mean(axis=0))
    return env

def modulation_energy(env, sr_hop, f_lo, f_hi):
    # env sampled at frame_rate = sr / hop_len
    T = len(env)
    # remove DC
    env = env - env.mean()
    spec = np.fft.rfft(env * np.hanning(T))
    freqs = np.fft.rfftfreq(T, d=1.0/sr_hop)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    power = (np.abs(spec)**2)[band].sum()
    return float(power)

def tnr_from_harmonics(mel, mel_edges, f0_track, sr, tol=HARM_BW, max_harm=MAX_HARM):
    """
    Tonal-to-noise proxy:
    Ratio of energy inside harmonic masks to total energy in 200-3000 Hz mel bands.
    """
    centers = hz_to_mel_freq_centers(mel_edges)
    band_mask = (centers >= ENV_BAND[0]) & (centers <= ENV_BAND[1])
    mel_band = mel[band_mask]  # (B, T)
    total = mel_band.sum(axis=0) + 1e-9

    B, T = mel_band.shape
    harm_energy = np.zeros(T, dtype=np.float32)
    for t in range(T):
        f0 = f0_track[t]
        if f0 <= 0: 
            continue
        # collect harmonic bins
        mask = np.zeros(B, dtype=bool)
        for k in range(1, max_harm+1):
            fk = f0 * k
            lo = fk * (1 - tol); hi = fk * (1 + tol)
            mask |= ((centers[band_mask] >= lo) & (centers[band_mask] <= hi))
        harm_energy[t] = mel_band[mask, t].sum()
    tnr = harm_energy / total
    return tnr  # 0..1

def frame_hop_len(sr):
    hop_len = int(round(HOP_S * sr))
    # make hop a power-of-two divisor if you like; not necessary here
    return max(1, hop_len)

def main():
    ap = argparse.ArgumentParser(description="Extract UAV features (mel + physics cues) to NPY/CSV")
    ap.add_argument("wav", help="Path to WAV")
    ap.add_argument("--outdir", default=None, help="Output directory (default: alongside WAV)")
    args = ap.parse_args()

    wav = Path(args.wav)
    y, sr = sf.read(str(wav), dtype="float32")
    if y.ndim > 1: y = y.mean(axis=1)
    y, sr = resample_if_needed(y, sr)

    hop_len = frame_hop_len(sr)
    frame_rate = sr / hop_len

    # Features
    mel_db, mel_edges = compute_logmel(y, sr)             # (M, T)
    f0 = yin_f0(y, sr, hop_len)                           # (T,)
    smooth_N = max(1, int(SMOOTH_MS/1000.0 * frame_rate))
    f0_s = moving_avg(f0, smooth_N).astype(np.float32)    # smoothed f0
    flux = spectral_flux(mel_db)                          # (T,)
    tnr  = tnr_from_harmonics(mel_db, mel_edges, f0_s, sr)
    # envelope + modulation energy per BLOCK_S
    env = band_envelope(y, sr, *ENV_BAND)
    # Per-second aggregation
    block = int(round(BLOCK_S * frame_rate))
    T = mel_db.shape[1]
    n_blocks = int(np.ceil(T / block))

    # f0 variance per block
    f0_var = np.array([np.var(f0_s[i*block:(i+1)*block]) for i in range(n_blocks)], dtype=np.float32)
    flux_mean = np.array([np.mean(flux[i*block:(i+1)*block]) for i in range(n_blocks)], dtype=np.float32)
    tnr_mean  = np.array([np.mean(tnr[i*block:(i+1)*block])  for i in range(n_blocks)], dtype=np.float32)

    # modulation energy per block
    # env has same T as mel frames (via mel computation inside)
    modE = []
    for i in range(n_blocks):
        seg = env[i*block:(i+1)*block]
        if len(seg) < 4:
            modE.append(0.0)
        else:
            modE.append(modulation_energy(seg, frame_rate, MOD_LO_HZ, MOD_HI_HZ))
    modE = np.array(modE, dtype=np.float32)

    # Save arrays
    outdir = Path(args.outdir) if args.outdir else wav.parent
    outdir.mkdir(parents=True, exist_ok=True)

    np.save(outdir / (wav.stem + ".mel_db.npy"), mel_db)
    np.save(outdir / (wav.stem + ".f0.npy"), f0)
    np.save(outdir / (wav.stem + ".f0_smooth.npy"), f0_s)
    np.save(outdir / (wav.stem + ".flux.npy"), flux)
    np.save(outdir / (wav.stem + ".tnr.npy"), tnr)

    # Per-second CSV (one row per ~1 s)
    csv_path = outdir / (wav.stem + ".persec.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_s","end_s","f0_var","flux_mean","tnr_mean","mod_energy"])
        for i in range(n_blocks):
            s = i*BLOCK_S
            e = (i+1)*BLOCK_S
            w.writerow([f"{s:.2f}", f"{e:.2f}", f"{f0_var[i]:.6f}",
                        f"{flux_mean[i]:.6f}", f"{tnr_mean[i]:.6f}", f"{modE[i]:.6f}"])

    print("Saved:")
    print(" ", outdir / (wav.stem + ".mel_db.npy"))
    print(" ", outdir / (wav.stem + ".f0.npy"))
    print(" ", outdir / (wav.stem + ".f0_smooth.npy"))
    print(" ", outdir / (wav.stem + ".flux.npy"))
    print(" ", outdir / (wav.stem + ".tnr.npy"))
    print(" ", csv_path)

if __name__ == "__main__":
    main()
