import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# ✅ Base path
base_path = r"dataset"

train_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True)

train_generators = []
val_generators = []

# Load all datasets
for ds in datasets:
    train_path = f"{base_path}/{ds}/train"
    val_path = f"{base_path}/{ds}/validation"

    train_gen = train_datagen.flow_from_directory(
        train_path,
        target_size=(224,224),
        batch_size=32,
        class_mode='binary'
    )

    val_gen = val_datagen.flow_from_directory(
        val_path,
        target_size=(224,224),
        batch_size=32,
        class_mode='binary'
    )

    train_generators.append(train_gen)
    val_generators.append(val_gen)

# Combine generators
def combined_generator(generators):
    while True:
        for gen in generators:
            for batch in gen:
                yield batch

train_gen = combined_generator(train_generators)
val_gen = combined_generator(val_generators)

# Count steps
train_steps = sum([g.samples for g in train_generators]) // 32
val_steps = sum([g.samples for g in val_generators]) // 32

# Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Fine-tuning
for layer in base_model.layers[-30:]:
    layer.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
 

# 🚀 TRAIN (keep 1 epoch for speed)
model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=1
)

# Save model
model.save("real_fake_model.h5")

print("✅ TRAINING DONE USING ALL 4 DATASETS")