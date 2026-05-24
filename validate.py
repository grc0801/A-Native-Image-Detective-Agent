import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================
# 🔹 CONFIG
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = r"D:\Fake_or_Real\real_fake_model.h5"

# =========================
# 🔹 LOAD MODEL
# =========================
model = load_model(MODEL_PATH)

# =========================
# 🔹 DATA GENERATOR
# =========================
datagen = ImageDataGenerator(rescale=1./255)

# =========================
# 🔹 DATASET PATHS
# =========================
TEST_DIRS = [
    "dataset/dataset1/test/real",
    "dataset/dataset2/test/real",
    "dataset/dataset3/test/real",
    "dataset/dataset4/test/real"
]

all_preds = []
all_labels = []

# =========================
# 🔹 LOOP THROUGH DATASETS
# =========================
for path in TEST_DIRS:
    print(f"\nProcessing: {path}")

    test_gen = datagen.flow_from_directory(
        path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    pred_probs = model.predict(test_gen)
    preds = (pred_probs > 0.5).astype(int).flatten()

    all_preds.extend(preds)
    all_labels.extend(test_gen.classes)

# =========================
# 🔹 FINAL RESULTS
# =========================
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds))

accuracy = np.mean(all_preds == all_labels)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")