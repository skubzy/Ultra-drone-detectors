import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import sounddevice as sd
import soundfile as sf
import csv
import time

def record_clip(seconds: float, sr: int, channels: int, device=None):
    if device is not None:
        sd.default.device = device
    print(f"  ▶ Recording {seconds}s @ {sr} Hz...")
    rec = sd.rec(int(seconds * sr), samplerate=sr, channels=channels, dtype="float32")
    sd.wait()
    return rec

def main():
    ap = argparse.ArgumentParser(description="Record WAV clips for dataset collection.")
    ap.add_argument("--outdir", required=True, help="Output directory for WAV files")
    ap.add_argument("--label", default="", help="Optional label (subfolder) e.g., fan, noise, drone")
    ap.add_argument("--sr", type=int, default=32000, help="Sample rate (Hz), default 32000")
    ap.add_argument("--dur", type=float, default=5.0, help="Clip duration (seconds), default 5.0")
    ap.add_argument("--clips", type=int, default=10, help="How many clips to record, default 10")
    ap.add_argument("--prefix", default="rec", help="Filename prefix, default 'rec'")
    ap.add_argument("--channels", type=int, default=1, help="Channels, default 1 (mono)")
    ap.add_argument("--device", type=int, default=None, help="Input device index (optional). Use --list to see devices.")
    ap.add_argument("--gap", type=float, default=1.0, help="Gap between clips (seconds), default 1.0")
    ap.add_argument("--list", action="store_true", help="List audio devices and exit")
    args = ap.parse_args()

    if args.list:
        print(sd.query_devices())
        return

    # Prepare output directory (optionally with label subfolder)
    outdir = Path(args.outdir)
    if args.label:
        outdir = outdir / args.label
    outdir.mkdir(parents=True, exist_ok=True)

    meta_path = outdir / "metadata.csv"
    write_header = not meta_path.exists()

    print(f"\nSaving WAV clips to: {outdir.resolve()}")
    if args.device is not None:
        print(f"Using input device index: {args.device}")

    try:
        with open(meta_path, "a", newline="") as fmeta:
            writer = csv.writer(fmeta)
            if write_header:
                writer.writerow(["filename", "label", "sr", "channels", "duration_s", "timestamp"])

            for i in range(args.clips):
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                fname = f"{args.prefix}_{ts}_{i:03d}.wav"
                wav_path = outdir / fname

                # Record
                audio = record_clip(args.dur, args.sr, args.channels, device=args.device)

                # Write as standard PCM 16-bit WAV (widely compatible)
                sf.write(wav_path, audio, args.sr, subtype="PCM_16")

                # Log metadata
                writer.writerow([fname, args.label, args.sr, args.channels, args.dur, ts])
                print(f"  💾 Saved: {wav_path.name}")

                # Small gap between takes
                if i < args.clips - 1 and args.gap > 0:
                    time.sleep(args.gap)

        print("\nDone. Happy training!")
        print(f"Metadata: {meta_path.resolve()}")

    except KeyboardInterrupt:
        print("\nStopped by user. Partial dataset saved.")
    except Exception as e:
        print("Error:", e)
        print("Tip: On Windows/macOS, ensure microphone permission is enabled for your terminal/IDE.")
        print("Tip: To pick a device, run with --list, note the input index, then pass --device <index>.")

if __name__ == "__main__":
    main()
