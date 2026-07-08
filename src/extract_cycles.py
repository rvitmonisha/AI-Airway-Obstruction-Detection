import os
import pandas as pd
import librosa
import soundfile as sf

audio_folder = "datasets/raw_audio"
annotation_folder = "datasets/annotation"
output_folder = "outputs/breathing_cycles"

os.makedirs(output_folder, exist_ok=True)

audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

total_cycles = 0

for audio_file in audio_files:

    annotation_file = audio_file.replace(".wav", ".txt")

    annotation_path = os.path.join(annotation_folder, annotation_file)

    
    if not os.path.exists(annotation_path):
        print(f"Annotation not found for {audio_file}")
        continue

    audio_path = os.path.join(audio_folder, audio_file)

    audio, sr = librosa.load(audio_path, sr=None)

    df = pd.read_csv(
        annotation_path,
        sep="\t",
        header=None,
        names=["Start", "End", "Crackles", "Wheezes"]
    )

    for i, row in df.iterrows():

        start_sample = int(row["Start"] * sr)
        end_sample = int(row["End"] * sr)

        cycle = audio[start_sample:end_sample]

        filename = f"{audio_file[:-4]}_cycle_{i+1}.wav"

        sf.write(
            os.path.join(output_folder, filename),
            cycle,
            sr
        )

        total_cycles += 1

print("--------------------------------")
print("Finished Successfully!")
print(f"Total Audio Files Processed: {len(audio_files)}")
print(f"Total Breathing Cycles Extracted: {total_cycles}")