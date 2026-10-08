import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os, math


base_dir = r"D:\FPTU\Semester 5\DPL302m\datasets\Screw Datasets\screw_all_clean"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "valid")
test_dir = os.path.join(base_dir, "test")

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],
    shear_range=0.25,
    channel_shift_range=15,
)


val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load dataset
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(192, 192),
    batch_size=32,
    class_mode='categorical'
)
val_gen = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(192, 192),
    batch_size=32,
    class_mode='categorical'
)
test_gen = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(192, 192),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Transfer Learning: MobileNetV2
base_model = MobileNetV2(
    input_shape=(192, 192, 3),
    include_top=False,
    weights='imagenet'
)

# Thêm regularization cho Conv layer
for layer in base_model.layers:
    if isinstance(layer, layers.Conv2D):
        layer.kernel_regularizer = regularizers.l2(1e-3)

base_model.trainable = False

# Xây dựng mô hình
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

# Learning rate schedule (Cosine Decay)
initial_lr = 1e-3
def cosine_decay(epoch):
    return initial_lr * 0.5 * (1 + math.cos(epoch / 30 * math.pi))

lr_scheduler = LearningRateScheduler(cosine_decay)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=initial_lr),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(patience=6, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.3, patience=3, min_lr=1e-6),
    ModelCheckpoint("best_MobileNetv2_stage1.h5", save_best_only=True),
    lr_scheduler
]

# Giai đoạn 1: Feature Extraction
print("\n STAGE 1: FEATURE EXTRACTION")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks
)

# Giai đoạn 2: Fine-tuning (Gradual Unfreezing)
print("\n STAGE 2: FINE-TUNING")
base_model.trainable = True
for layer in base_model.layers[:-60]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4,weight_decay=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-7),
        ModelCheckpoint("best_MobileNetv2_finetune.h5", save_best_only=True)
    ]
)

# Đánh giá
print("\n MODEL EVALUATION")
test_loss, test_acc = model.evaluate(test_gen)
print(f"\n Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

model.save("sheet_metal_mobilenetv2_optimized.h5")

# Biểu đồ Accuracy & Loss
acc = history.history['accuracy'] + fine_tune_history.history['accuracy']
val_acc = history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
loss = history.history['loss'] + fine_tune_history.history['loss']
val_loss = history.history['val_loss'] + fine_tune_history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, acc, label='Train Accuracy', color='blue')
plt.plot(epochs, val_acc, label='Validation Accuracy', color='orange')
plt.plot(epochs, loss, label='Train Loss', color='green', linestyle='--')
plt.plot(epochs, val_loss, label='Validation Loss', color='red', linestyle='--')
plt.title('Training Progress (Accuracy & Loss)')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Confusion Matrix
y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes
labels = list(test_gen.class_indices.keys())

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print("\n CLASSIFICATION REPORT ")
print(classification_report(y_true, y_pred_classes, target_names=labels))

print("\n SUMMARY METRICS ")
print(f"Train Accuracy (last epoch): {acc[-1]:.4f}")
print(f"Val Accuracy (last epoch): {val_acc[-1]:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
