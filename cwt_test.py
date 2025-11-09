import numpy as np
import audioflux as af
from audioflux.type import SpectralFilterBankScaleType, WaveletContinueType
from audioflux.utils import note_to_hz

import matplotlib.pyplot as plt
from audioflux.display import fill_spec

# Get a 220Hz's audio file path
from pathlib import Path
sample_path = str(Path(r"C:\Users\Pc\Desktop\Desktop\Stuff\test123\mic_test.wav"))

# Read audio data and sample rate
audio_arr, sr = af.read(sample_path)
audio_arr = audio_arr[..., :4096]

cwt_obj = af.CWT(num=84, radix2_exp=12, samplate=sr, low_fre=note_to_hz('C1'),
                 bin_per_octave=12, wavelet_type=WaveletContinueType.MORSE,
                 scale_type=SpectralFilterBankScaleType.OCTAVE)

cwt_spec_arr = cwt_obj.cwt(audio_arr)

synsq_obj = af.Synsq(num=cwt_obj.num,
                     radix2_exp=cwt_obj.radix2_exp,
                     samplate=cwt_obj.samplate)

synsq_arr = synsq_obj.synsq(cwt_spec_arr,
                            filter_bank_type=cwt_obj.scale_type,
                            fre_arr=cwt_obj.get_fre_band_arr())

# Show CWT
fig, ax = plt.subplots(figsize=(7, 4))
img = fill_spec(np.abs(cwt_spec_arr), axes=ax,
                x_coords=cwt_obj.x_coords(),
                y_coords=cwt_obj.y_coords(),
                x_axis='time', y_axis='log',
                title='CWT')
fig.colorbar(img, ax=ax)
# Show Synsq
fig, ax = plt.subplots(figsize=(7, 4))
img = fill_spec(np.abs(synsq_arr), axes=ax,
                x_coords=cwt_obj.x_coords(),
                y_coords=cwt_obj.y_coords(),
                x_axis='time', y_axis='log',
                title='Synsq')
fig.colorbar(img, ax=ax)

plt.show()