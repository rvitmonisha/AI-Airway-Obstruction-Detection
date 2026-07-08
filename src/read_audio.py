import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

audio_folder = "datasets/raw_audio"

audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

print(f"Total audio files: {len(audio_files)}")

first_file = audio_files[0]
print("Reading:", first_file)

audio_path = os.path.join(audio_folder, first_file)

audio, sample_rate = librosa.load(audio_path, sr=None)

print("Sample Rate:", sample_rate)
print("Audio Shape:", audio.shape)
print("Duration:", len(audio) / sample_rate, "seconds")

plt.figure(figsize=(12, 4))
librosa.display.waveshow(audio, sr=sample_rate)

plt.title("Respiratory Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.show()