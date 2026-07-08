import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ==========================================
# 1. LOAD LABELS CSV
# ==========================================
labels_df = pd.read_csv("labels/labels.csv")

images = []
labels = []
spectrogram_folder = "outputs/spectrograms"

print("🔄 Reading and loading clean numeric spectrogram arrays (.npy)...")

# ==========================================
# 2. ITERATE AND RESOLVE FILE PATHS
# ==========================================
for _, row in labels_df.iterrows():
   
    array_name = row["Filename"].replace(".wav", ".npy")
    array_path = os.path.join(spectrogram_folder, array_name)

    if os.path.exists(array_path):
        matrix = np.load(array_path)
        
       
        target_width = 128
        if matrix.shape[1] < target_width:
            pad_width = target_width - matrix.shape[1]
            matrix = np.pad(matrix, ((0, 0), (0, pad_width)), mode='constant')
        elif matrix.shape[1] > target_width:
            matrix = matrix[:, :target_width]
            
        images.append(matrix)
        labels.append(row["Label"])


X_raw = np.array(images)
y_raw = np.array(labels)

if len(X_raw) == 0:
    raise ValueError(
        f"❌ Severe Error: Found 0 matching array files inside '{spectrogram_folder}'. "
        "Verify your extract_features.py generated valid .npy files."
    )

# ==========================================
# 3. FIX 1: SAMPLE-BY-SAMPLE NORMALIZATION 
# ==========================================
print("⚖️ Normalizing each spectrogram on an individual file level...")
X_normalized = []
for matrix in X_raw:
    
    norm_matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix) + 1e-8)
    X_normalized.append(norm_matrix)

X = np.array(X_normalized)

X = np.expand_dims(X, axis=-1)
y = y_raw

print(f"📊 Extracted Array Dimensions: {X.shape}")
print(f"📊 Target Categorical Label Size: {y.shape}")

# ==========================================
# 4. ENCODE LABELS
# ==========================================
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("\n🏷️ Target Class Mappings:")
for idx, label in enumerate(encoder.classes_):
    print(f"  Class Index {idx} -> Label: {label}")

# ==========================================
# 5. STRATIFIED DATASET SPLIT (80/20)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Splitting Statistics: {len(X_train)} Train Profiles | {len(X_test)} Test Profiles")

# ==========================================
# 6. FIX 2: COMPUTE BALANCED CLASS WEIGHTS
# ==========================================

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

print("\n⚖️ Computed Class Penalty Weights to eliminate 'Normal' guessing shortcuts:")
for class_idx, weight in class_weight_dict.items():
    print(f"  Class {encoder.classes_[class_idx]} Penalty Factor Weight multiplier: {weight:.4f}")

# ==========================================
# 7. BUILD CNN MODEL ARCHITECTURE
# ==========================================
model = Sequential([
    
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🖥️ CNN Network Topology:")
model.summary()

# ==========================================
# 8. EXECUTE OPTIMIZATION (TRAINING)
# ==========================================
print("\n🚀 Commencing model training epochs with class balancing configurations...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32,
    class_weight=class_weight_dict  
)

# ==========================================
# 9. EVALUATE & EXPORT MODEL WEIGHTS
# ==========================================
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n🎯 Realist Cross-Validation Accuracy Score: {accuracy * 100:.2f}%")

os.makedirs("models", exist_ok=True)
model.save("models/airway_cnn_model.h5")
print("\n💾 Model Saved Successfully to 'models/airway_cnn_model.h5'!")