import audioflux as af, numpy as np, matplotlib.pyplot as plt
from audioflux.display import fill_spec

wav_path = "briskaudioclip2.wav"  # put a real file here
audio, sr = af.read(wav_path)

spec, mel_freqs = af.mel_spectrogram(audio, num=128, radix2_exp=12, samplate=sr)
spec = 20*np.log10(np.abs(spec)+1e-12)

x = np.linspace(0, audio.shape[-1]/sr, spec.shape[-1]+1)
y = np.insert(mel_freqs, 0, 0)
fig, ax = plt.subplots()
img = fill_spec(spec, axes=ax, x_coords=x, y_coords=y, x_axis='time', y_axis='log',
                title='Mel Spectrogram (dB)')
fig.colorbar(img, ax=ax); plt.show()
