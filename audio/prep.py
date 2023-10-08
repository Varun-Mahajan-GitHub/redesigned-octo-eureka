import librosa,librosa.display
import matplotlib.pyplot as plt
import numpy as np

# waveform
file = 'Alesis-Fusion-Acoustic-Bass-C2.wav'
signal, sr = librosa.load(file)

librosa.display.waveshow(signal,sr=sr,x_axis='ms')
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# FFT -> frequency specturm
fft = np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0,sr,len(magnitude))

left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]

plt.plot(left_frequency,left_magnitude)
plt.show()

# stft -> spectrograph
stft = librosa.stft(signal,hop_length=512,n_fft=2048)
spectrogram = np.abs(stft)
librosa.display.specshow(spectrogram,sr=sr,hop_length=512)
plt.xlabel('Time')
plt.ylabel('frequency')
plt.colorbar()
plt.show()
# MFCCs