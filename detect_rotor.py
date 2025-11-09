import argparse, csv
from pathlib import Path
import numpy as np
import soundfile as sf
import audioflux as af
import matplotlib.pyplot as plt

# ----- defaults (tweak if needed) -----
SR_TARGET   = 32000
N_MELS      = 96
LOW_F, HI_F = 50.0, 8000.0
RADIX2_EXP  = 12          # FFT size 4096
HOP_S       = 0.010       # ~10 ms hop
MOD_BAND    = (10.0, 80.0)   # blade-pass AM band (Hz)
ENERGY_BAND = (200.0, 3000.0) # focus band for rotor energy
HARM_BW     = 0.08        # ±8% harmonic mask
MAX_HARM    = 8
# decision smoothing
MIN_ON_SEC  = 0.4
MIN_OFF_SEC = 0.3
THRESH      = 0.55        # rotor score -> 1 if >= THRESH
PLOT        = True
# --------------------------------------
def resample_to(y, sr, target):
    if sr == target:
        return y.astype(np.float32), sr
    try:
        y2 = af.resample(y, sr, target)   # positional args
    except (TypeError, AttributeError):
        n_new = int(round(len(y) * target / sr))
        x_old = np.linspace(0.0, 1.0, len(y), endpoint=False)
        x_new = np.linspace(0.0, 1.0, n_new, endpoint=False)
        y2 = np.interp(x_new, x_old, y).astype(np.float32)
        return y2, target
    return y2.astype(np.float32), target

def f0_simple(y, sr, hop_len, fmin=60, fmax=800):
    """
    Quick f0 estimator: per frame, pick the strongest peak in 60–800 Hz from STFT.
    Not as robust as YIN, but fine for rotor vs other.
    """
    import numpy as np
    win = 2**RADIX2_EXP
    step = hop_len
    n = len(y)
    hann = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, 1.0/sr)
    band = (freqs >= fmin) & (freqs <= fmax)
    f0 = []
    for i in range(0, n - win + 1, step):
        frame = y[i:i+win] * hann
        mag = np.abs(np.fft.rfft(frame))
        idx = band.nonzero()[0]
        if idx.size == 0:
            f0.append(0.0); continue
        peak_i = idx[np.argmax(mag[band])]
        f0.append(float(freqs[peak_i]))
    return np.array(f0, dtype=np.float32)


def logmel(y, sr):
    spec, edges = af.mel_spectrogram(y, num=N_MELS, radix2_exp=RADIX2_EXP,
                                     samplate=sr, low_fre=LOW_F, high_fre=HI_F)
    mel = np.abs(spec).astype(np.float32) + 1e-12
    mel_db = 20*np.log10(mel)
    return mel_db, edges

def mel_centers(edges):
    return 0.5*(edges[:-1]+edges[1:])

def spectral_flatness(mel):
    # mel: (M,T) -> flatness per frame
    # SFM = exp(mean(log(x)))/mean(x)
    x = np.maximum(mel, 1e-12)
    gm = np.exp(np.mean(np.log(x), axis=0))
    am = np.mean(x, axis=0)
    return (gm/(am+1e-12)).astype(np.float32)

def spectral_flux(mel_db):
    d = np.diff(mel_db, axis=1)
    d[d<0] = 0
    flux = d.sum(axis=0)
    flux = np.concatenate([[flux[0]], flux])
    return flux.astype(np.float32)

def yin_f0(y, sr, hop_len):
    """
    Returns f0 (Hz) per frame. Works across audioflux versions.
    """
    import numpy as np
    try:
        # Newer API path (if it supports slide_length in __init__)
        pitch = af.PitchYIN(samplate=sr, slide_length=hop_len)
        f0 = pitch.pitch(y)
    except TypeError:
        # Older API: pass radix2_exp / slide_length to .pitch()
        pitch = af.PitchYIN(samplate=sr)
        f0 = pitch.pitch(y, radix2_exp=RADIX2_EXP, slide_length=hop_len)
    f0 = np.asarray(f0, dtype=np.float32)
    f0[f0 < 1.0] = 0.0
    return f0


def modulation_energy_from_env(env, fps, lo, hi):
    T = len(env)
    if T < 8: return 0.0
    env = env - env.mean()
    win = np.hanning(T)
    spec = np.fft.rfft(env*win)
    freqs = np.fft.rfftfreq(T, d=1.0/fps)
    mask = (freqs>=lo)&(freqs<=hi)
    return float((np.abs(spec)**2)[mask].sum())

def band_envelope(mel, edges, lo, hi):
    c = mel_centers(edges)
    mask = (c>=lo)&(c<=hi)
    env = np.sqrt((np.maximum(mel[mask],1e-12)**2).mean(axis=0))
    return env.astype(np.float32)

def harmonic_mask_energy(mel, edges, f0, lo, hi, harm_bw=0.08, max_harm=8):
    c = mel_centers(edges)
    band = (c>=lo)&(c<=hi)
    melB = mel[band]                 # (B,T)
    cB   = c[band]
    tot  = melB.sum(axis=0)+1e-9
    T = mel.shape[1]
    H = np.zeros(T, dtype=np.float32)
    for t in range(T):
        f = f0[t]
        if f<=0: continue
        m = np.zeros_like(cB, dtype=bool)
        for k in range(1, max_harm+1):
            fk = f*k
            m |= (cB >= fk*(1-harm_bw)) & (cB <= fk*(1+harm_bw))
        H[t] = melB[m, t].sum()
    return (H/tot).astype(np.float32)  # 0..1

def smooth_bool(seq, on_len, off_len, fps):
    on_N  = max(1, int(round(on_len*fps)))
    off_N = max(1, int(round(off_len*fps)))
    out = np.zeros_like(seq, dtype=bool)
    state=False; run=0
    for i,x in enumerate(seq):
        if x and not state:
            run+=1
            if run>=on_N: state=True; run=0
        elif not x and state:
            run+=1
            if run>=off_N: state=False; run=0
        else:
            run=0
        out[i]=state
    return out

def main():
    ap = argparse.ArgumentParser(description="Rotor (fan/drone-like) vs other detector")
    ap.add_argument("wav", help="Path to WAV")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    wav = Path(args.wav)
    outdir = Path(args.outdir) if args.outdir else wav.parent
    outdir.mkdir(parents=True, exist_ok=True)

    y, sr = sf.read(str(wav), dtype="float32")
    if y.ndim>1: y = y.mean(axis=1)
    y, sr = resample_to(y, sr, SR_TARGET)

    hop = max(1, int(round(HOP_S*sr)))
    fps = sr / hop
    dur = len(y)/sr

    # features
    mel_db, edges = logmel(y, sr)                      # (M,T)
    mel_lin = np.maximum(10**(mel_db/20.0), 1e-12)     # back to linear for some measures
    f0 = f0_simple(y, sr, hop)                            # (T,)
    sfm = spectral_flatness(mel_lin)                   # (T,)
    flux = spectral_flux(mel_db)                       # (T,)
    env = band_envelope(mel_lin, edges, *ENERGY_BAND)  # (T,)
    modE = []
    block = int(round(1.0*fps))                        # 1s blocks
    for i in range(int(np.ceil(len(env)/block))):
        seg = env[i*block:(i+1)*block]
        modE.append(modulation_energy_from_env(seg, fps, *MOD_BAND))
    # expand modE to frame rate
    modE = np.repeat(np.array(modE, dtype=np.float32), block)[:len(env)]
    tnr = harmonic_mask_energy(mel_lin, edges, f0, *ENERGY_BAND,
                               harm_bw=HARM_BW, max_harm=MAX_HARM)  # 0..1

    # normalize helper (robust)
    def rn(x):
        p5, p95 = np.percentile(x,5), np.percentile(x,95)
        return np.clip((x-p5)/(p95-p5+1e-9), 0, 1).astype(np.float32)

    # build a simple rotor score (0..1):
    # high when: strong harmonics (tnr↑), tonal (flatness↓), modulation present (modE↑), moderate change (flux↑ a bit)
    score = 0.45*rn(tnr) + 0.25*(1.0 - rn(sfm)) + 0.20*rn(modE) + 0.10*rn(flux)

    # binary decision with hysteresis
    raw = score >= THRESH
    mask = smooth_bool(raw, MIN_ON_SEC, MIN_OFF_SEC, fps)

    # export segments
    segments=[]
    in_on=False; start=0.0
    for i,m in enumerate(mask):
        t = i/fps
        if m and not in_on:
            in_on=True; start=t
        elif (not m) and in_on:
            in_on=False; segments.append((start, t))
    if in_on:
        segments.append((start, len(mask)/fps))

    # save per-frame score
    np.save(outdir / (wav.stem + ".rotor_score.npy"), score)

    # save segments CSV
    csv_path = outdir / (wav.stem + ".rotor_segments.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_s","end_s"])
        for s,e in segments:
            w.writerow([f"{s:.3f}", f"{e:.3f}"])

    print("Saved:")
    print(" ", outdir / (wav.stem + ".rotor_score.npy"))
    print(" ", csv_path)

    if PLOT and not args.no_plot:
        T = len(score); t = np.arange(T)/fps
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(t, score, lw=1.5, label="rotor score")
        ax.axhline(THRESH, ls="--", color="k", lw=1.0, label=f"threshold {THRESH}")
        for s,e in segments:
            ax.axvspan(s, e, color="tab:green", alpha=0.2)
        ax.set_ylim(0,1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Score")
        ax.set_title("Rotor vs Other — decision timeline")
        ax.legend()
        plt.tight_layout()
        fig.savefig(outdir / (wav.stem + ".rotor_timeline.png"), dpi=200)
        plt.show()

if __name__ == "__main__":
    main()
