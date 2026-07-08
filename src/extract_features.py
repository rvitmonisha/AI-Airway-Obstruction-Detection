import os
import librosa
import numpy as np

input_folder = "outputs/breathing_cycles"
spectrogram_folder = "outputs/spectrograms"  
mfcc_folder = "outputs/mfcc"

os.makedirs(spectrogram_folder, exist_ok=True)
os.makedirs(mfcc_folder, exist_ok=True)

audio_files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]

print("--------------------------------")
print(f"Total breathing cycles found: {len(audio_files)}")
print("Starting feature extraction pipeline...")
print("--------------------------------")

processed_count = 0

for file in audio_files:
    audio_path = os.path.join(input_folder, file)
    
    try:
       
        signal, sr = librosa.load(audio_path, sr=None)

        
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=13
        )
        
        
        np.save(
            os.path.join(mfcc_folder, file.replace(".wav", ".npy")),
            mfcc
        )

        
        mel = librosa.feature.melspectrogram(
            y=signal,
            sr=sr,
            n_mels=128
        )

        
        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        
        np.save(
            os.path.join(spectrogram_folder, file.replace(".wav", ".npy")),
            mel_db
        )
        
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Progress: Processed {processed_count}/{len(audio_files)} cycles.")

    except Exception as e:
        print(f" Error processing file {file}: {str(e)}")

print("--------------------------------")
print("Feature extraction completed successfully!")
print(f"Clean numeric data arrays saved to '{spectrogram_folder}' and '{mfcc_folder}'.")