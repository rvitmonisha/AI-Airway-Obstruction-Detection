import os
import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.models import load_model


LABELS = ["Both", "Crackles", "Normal", "Wheezes"]

# ==========================================
# 1. LOAD MODEL ARCHITECTURE
# ==========================================
model_path = "models/airway_cnn_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Trained brain missing at '{model_path}'. Run train_cnn.py first.")

model = load_model(model_path)
print("✅ Diagnostic Model Weights Loaded Successfully!")


audio_path = "test_audio/103_2b2_Ar_mc_LittC2SE.wav"
if not os.path.exists(audio_path):
   
    audio_path = "datasets/raw_audio/104_1b1_Ar_sc_Litt3200.wav"

print(f"\n🎵 Ingesting Target Patient Audio: {os.path.basename(audio_path)}")

audio_signal, sr = librosa.load(audio_path, sr=None)
duration = len(audio_signal) / sr
print(f"⏱️ Sample Rate: {sr} Hz | Total Signal Duration: {duration:.2f} seconds")

# ==========================================
# 2. HELPER METRIC FUNCTION FOR SINGLE VECTOR
# ==========================================
def process_and_classify_segment(audio_segment, sample_rate):
    """Processes a single isolated breath matrix exactly like the training data."""
    
    mel = librosa.feature.melspectrogram(y=audio_segment, sr=sample_rate, n_mels=128)
    matrix = librosa.power_to_db(mel, ref=np.max)
    
    
    target_width = 128
    if matrix.shape[1] < target_width:
        pad_width = target_width - matrix.shape[1]
        matrix = np.pad(matrix, ((0, 0), (0, pad_width)), mode='constant')
    elif matrix.shape[1] > target_width:
        matrix = matrix[:, :target_width]
        
    matrix_scaled = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix) + 1e-8)
    
    
    input_tensor = np.expand_dims(matrix_scaled, axis=(0, -1))
    
    
    predictions = model.predict(input_tensor, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100
    
    return LABELS[class_idx], confidence

# ==========================================
# 3. INTERSECT ANNOTATION TIMES OR RUN SNAPSHOTS
# ==========================================
base_name = os.path.splitext(os.path.basename(audio_path))[0]
annotation_path = os.path.join("datasets/annotation", f"{base_name}.txt")

print("\n--- 🎯 REAL-TIME DIAGNOSTIC ASSESSMENT BREAKDOWN ---")

if os.path.exists(annotation_path):
    print(f"📋 Annotation guide found at '{annotation_path}'. Parsing designated breathing cycles...")
    
    df = pd.read_csv(annotation_path, sep="\t", header=None, names=["Start", "End", "Crackles", "Wheezes"])
    
    for i, row in df.iterrows():
        start_sample = int(row["Start"] * sr)
        end_sample = int(row["End"] * sr)
        
        
        cycle_segment = audio_signal[start_sample:end_sample]
        
        if len(cycle_segment) > 0:
            condition, conf = process_and_classify_segment(cycle_segment, sr)
            print(f" Cycle #{i+1} ({row['Start']:.1f}s - {row['End']:.1f}s) ➔ Classified: {condition:<10} | Confidence: {conf:.2f}%")

else:
    print("⚠️ No structural time marker annotations found. Splitting audio using 3-second sliding window snapshots...")
    
    window_size = int(3.0 * sr)
    step_size = int(2.5 * sr)
    start_idx = 0
    snapshot_count = 1
    
    while start_idx + window_size <= len(audio_signal):
        segment = audio_signal[start_idx : start_idx + window_size]
        condition, conf = process_and_classify_segment(segment, sr)
        
        time_start = start_idx / sr
        time_end = (start_idx + window_size) / sr
        print(f" Snapshot #{snapshot_count} ({time_start:.1f}s - {time_end:.1f}s) ➔ Classified: {condition:<10} | Confidence: {conf:.2f}%")
        
        start_idx += step_size
        snapshot_count += 1

print("----------------------------------------------------")