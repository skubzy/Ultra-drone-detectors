# Ultra-Drone-Detectors

An audio-based drone detection system that identifies unmanned aerial vehicles (UAVs) by analyzing acoustic signatures, particularly rotor blade sounds.

## Overview

This project uses machine learning and audio signal processing to detect and classify drone sounds from ambient audio. It extracts relevant features from audio signals (mel-spectrograms, MFCC, harmonic analysis) and applies detection algorithms to identify rotor blade characteristics.

## Features

- **Audio Feature Extraction**: Computes mel-spectrograms, MFCC, and harmonic analysis from WAV files
- **Rotor Detection**: Identifies drone rotor sounds based on acoustic signatures
- **Dataset Recording**: Tools to record and organize audio datasets for training and testing
- **Multi-Format Support**: Processes .wav audio files and numpy arrays (.npy)
- **Real-time Analysis**: Live audio spectrogram analysis capabilities

## Project Structure

```
Ultra-drone-detectors/
├── extract_uav_features.py      # Feature extraction (MEL, MFCC, harmonics)
├── detect_rotor.py              # Rotor/blade detection algorithm
├── dataset_recorder.py          # Record audio clips for dataset creation
├── live_mel_spectrogram.py      # Real-time mel-spectrogram visualization
├── live_mel_debug.py            # Debug live mel-spectrograms
├── make_mfcc_39.py              # Generate 39-dim MFCC features
├── make_specs.py                # Generate spectrograms
├── tile_maker.py                # Create tiled datasets
├── cwt_test.py                  # Continuous wavelet transform tests
└── data/
    ├── drone/                   # Drone audio samples and extracted features
    │   ├── *_audio.npy         # Raw audio waveforms
    │   ├── *_mel_db.npy        # Mel-spectrogram (dB scale)
    │   ├── *_mfcc.npy          # MFCC features
    │   └── *_mel/              # Mel-spectrogram frames
    ├── non-drone/              # Non-drone audio samples and features
    └── UAVDataset/
        └── fan/
            ├── Clear-Drone/
            ├── Clear-Nodrone/
            ├── Noisy-Drone/
            └── Noisy-Nodrone/
```

## Dependencies

- **numpy**: Numerical computing
- **soundfile**: Audio I/O
- **sounddevice**: Real-time audio recording
- **audioflux**: Advanced audio signal processing (spectrograms, resampling, mel-filters)
- **matplotlib**: Plotting and visualization

## Quick Start

### 1. Record Audio Dataset

```bash
python dataset_recorder.py --outdir data/drone --label mydrone --sr 32000 --dur 5 --clips 10
```

Options:
- `--outdir`: Output directory for WAV files
- `--label`: Optional subfolder label (e.g., drone, non-drone)
- `--sr`: Sample rate (default: 32000 Hz)
- `--dur`: Clip duration in seconds (default: 5.0)
- `--clips`: Number of clips to record (default: 10)
- `--device`: Audio device index (use `--list` to see available devices)

### 2. Extract Audio Features

```bash
python extract_uav_features.py --infile audio.wav --outdir features/
```

Extracts:
- Mel-spectrograms (128 bands, log scale)
- MFCC features
- Harmonic/fundamental frequency information
- Modulation/envelope analysis

### 3. Detect Rotor Sounds

```bash
python detect_rotor.py --infile audio.wav --outdir results/
```

Generates:
- Rotor presence/absence classification
- Temporal detection scores
- Optional visualization plots

### 4. Generate MFCC Features

```bash
python make_mfcc_39.py --infile audio.wav --outdir mfcc_features/
```

Creates 39-dimensional MFCC feature vectors.

## Audio Processing Parameters

### Default Configuration
- **Sample Rate**: 32,000 Hz
- **Mel Bands**: 96–128 depending on script
- **Frequency Range**: 50–8,000 Hz
- **FFT Size**: 4,096 (2^12)
- **Hop Length**: ~10 ms (frame rate ~100 Hz)
- **Rotor Energy Band**: 200–3,000 Hz
- **Modulation Band (Blade-Pass)**: 10–80 Hz

### Key Thresholds
- **Rotor Detection Threshold**: 0.55 (confidence score)
- **Harmonic Bandwidth**: ±8% around fundamental
- **Minimum On Duration**: 0.4 seconds
- **Minimum Off Duration**: 0.3 seconds

## Data Format

### Input
- WAV audio files (mono or multi-channel)

### Output (Numpy arrays, .npy)
- `*_audio.npy`: Raw waveform samples (1D array)
- `*_mel_db.npy`: Mel-spectrogram in dB scale (2D: time × mel-freq)
- `*_mfcc.npy`: MFCC coefficients (2D: time × coefficients)
- `*_mel/`: Directory of individual mel-spectrogram frames (2D images)

## Rotor Detection Algorithm

The detection pipeline:
1. **Resample** audio to 32 kHz
2. **Compute mel-spectrogram** (96 bands, log scale)
3. **Estimate fundamental frequency (f0)** from spectrogram
4. **Identify harmonics** (up to 8 multiples of f0)
5. **Measure amplitude modulation** in rotor frequency band (10–80 Hz)
6. **Compute rotor score** based on harmonic stability and modulation
7. **Smooth temporal decisions** (0.4s minimum on, 0.3s minimum off)
8. **Output binary classification**: Rotor present (≥0.55) or absent (<0.55)

## Dataset Organization

For optimal results, organize datasets as:

```
UAVDataset/
├── Clear-Drone/     → Clean audio with drones present
├── Clear-Nodrone/   → Clean audio without drones
├── Noisy-Drone/     → Noisy environment with drones
└── Noisy-Nodrone/   → Noisy environment without drones
```

Each subdirectory should contain:
- `.wav` audio files
- Generated features (`.npy` files)
- `metadata.csv` with recording information

## Usage Examples

### Process an entire directory

```bash
for f in data/drone/*.wav; do
    python extract_uav_features.py --infile "$f" --outdir data/features/
done
```

### Batch rotor detection

```bash
for f in data/*.wav; do
    python detect_rotor.py --infile "$f" --outdir results/
done
```

## Troubleshooting

- **Audio device not found**: Use `python dataset_recorder.py --list` to see available devices, then specify with `--device`
- **Feature extraction fails**: Ensure input WAV files are valid and sample rates are supported (typically 16–48 kHz)
- **Poor rotor detection**: Adjust `THRESH` parameter in `detect_rotor.py` based on your audio characteristics

## Performance Considerations

- Processing time depends on audio duration and feature complexity
- Mel-spectrogram computation is the bottleneck for long audio files
- Real-time processing is possible for sample rates up to 48 kHz

## References

- **Mel-Spectrogram**: Maps frequency to mel-scale (human perception)
- **MFCC**: Mel-Frequency Cepstral Coefficients (commonly used in audio ML)
- **Harmonic Analysis**: Rotor blades produce periodic, harmonic-rich signals
- **Amplitude Modulation**: Blade-pass frequency modulates the rotor sound (~10–80 Hz for typical drones)

## Future Improvements

- Real-time streaming detection
- Deep learning classifier integration
- Multi-drone simultaneous detection
- Directional audio analysis (beamforming)
- Cross-platform GUI application

## License

[Specify your license here]

## Contact

For questions or issues, please contact [your email or GitHub profile].
