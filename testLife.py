import sounddevice as sd, numpy as np
from scipy.io.wavfile import write

sr = 32000
dur = 30  # seconds
print("Recording...")
rec = sd.rec(int(dur*sr), samplerate=sr, channels=1, dtype='float32')
sd.wait()
write("mic_test.wav", sr, rec)
print("Saved mic_test.wav")
