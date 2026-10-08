import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler

from sklearn.metrics import confusion_matrix, classification_report

# ---------- CONFIG ----------
base_dir = r"D:\FPTU\Semester 5\DPL302m\datasets\Sheet Metal Datasets\sheet_metal_all"
train_dir = os.path.join(base_dir, "train")
val_dir   = os.path.join(base_dir, "valid")
test_dir  = os.path.join(base_dir, "test")

IMAGE_SIZE = 160
BATCH_SIZE = 32
INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-5

# ---------- Data generators (force RGB) ----------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],
    shear_range=0.2,
    channel_shift_range=10,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

# IMPORTANT: force color_mode='rgb' so images are 3-channel
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

test_gen = val_datagen.flow_from_directory(
    test_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

print("Train classes:", train_gen.class_indices)
num_classes = train_gen.num_classes

# check image shape channel dimension
img_shape = train_gen.image_shape  # (height, width, channels)
print("Generator image shape:", img_shape)

# ---------- Decide whether to use imagenet weights ----------
use_imagenet = (img_shape[2] == 3)

weights_arg = 'imagenet' if use_imagenet else None
print("Will try to load EfficientNetB4 weights =", weights_arg)

# ---------- Build backbone with safe fallback ----------
def build_backbone_safe(input_shape, weights):
    try:
        base = EfficientNetB4(input_shape=input_shape, include_top=False, weights=weights)
        print("Loaded EfficientNetB4 with weights =", weights)
        return base
    except Exception as e:
        print("Warning: failed to load EfficientNetB4 with weights=", weights, "->", str(e))
        print("Falling back to EfficientNetB4 with weights=None (random init).")
        base = EfficientNetB4(input_shape=input_shape, include_top=False, weights=None)
        return base

input_shape = (IMAGE_SIZE, IMAGE_SIZE, img_shape[2])
base_model = build_backbone_safe(input_shape=input_shape, weights=weights_arg)

# add small kernel regularizer to convs if you want (optional)
for layer in base_model.layers:
    if isinstance(layer, layers.Conv2D):
        # safe set attribute (no re-compilation here, but harmless)
        try:
            layer.kernel_regularizer = regularizers.l2(1e-4)
        except Exception:
            pass

# freeze for feature extraction
base_model.trainable = False

# ---------- Build final model ----------
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax')
])

# ---------- LR schedule ----------
def cosine_decay(epoch):
    return INITIAL_LR * 0.5 * (1 + math.cos(epoch / 30 * math.pi))

lr_scheduler = LearningRateScheduler(cosine_decay)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_stage1 = [
    EarlyStopping(patience=6, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(factor=0.3, patience=3, min_lr=1e-6, monitor='val_loss'),
    ModelCheckpoint("best_model_stage1.h5", save_best_only=True, monitor='val_accuracy'),
    lr_scheduler
]

# ---------- Stage 1: feature extraction ----------
print("\nSTAGE 1: Feature extraction")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks_stage1
)

# ---------- Stage 2: fine-tuning ----------
print("\nSTAGE 2: Fine-tuning")
base_model.trainable = True
# freeze most layers, unfreeze top blocks
fine_tune_at = max(0, len(base_model.layers) - 30)
for i, layer in enumerate(base_model.layers):
    layer.trainable = (i >= fine_tune_at)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_stage2 = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-7, monitor='val_loss'),
    ModelCheckpoint("best_model_finetune.h5", save_best_only=True, monitor='val_accuracy')
]

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks_stage2
)

# ---------- Evaluate ----------
test_loss, test_acc = model.evaluate(test_gen)
print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

model.save("sheet_metal_efficientnetb4_optimized.h5")
print("Saved model to sheet_metal_efficientnetb4_optimized.h5")

# ---------- Plot combined metrics ----------
acc  = history1.history.get('accuracy', []) + history2.history.get('accuracy', [])
val_acc = history1.history.get('val_accuracy', []) + history2.history.get('val_accuracy', [])
loss = history1.history.get('loss', []) + history2.history.get('loss', [])
val_loss = history1.history.get('val_loss', []) + history2.history.get('val_loss', [])
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(10,6))
plt.plot(epochs, acc, label='Train Accuracy')
plt.plot(epochs, val_acc, label='Val Accuracy')
plt.plot(epochs, loss, label='Train Loss', linestyle='--')
plt.plot(epochs, val_loss, label='Val Loss', linestyle='--')
plt.legend(); plt.grid(True); plt.show()

# ---------- Confusion matrix & report ----------
y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes
labels = list(test_gen.class_indices.keys())

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.show()

print(classification_report(y_true, y_pred_classes, target_names=labels))
