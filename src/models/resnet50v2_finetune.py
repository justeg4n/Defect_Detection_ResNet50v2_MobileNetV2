import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os, math


base_dir = r"D:\FPTU\Semester 5\DPL302m\datasets\Screw Datasets\screw_all_clean"
train_dir = os.path.join(base_dir, "train")
val_dir   = os.path.join(base_dir, "valid")
test_dir  = os.path.join(base_dir, "test")

# Data augmentation và normalize
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],
    shear_range=0.2,
    channel_shift_range=10
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load dataset
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
val_gen = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
test_gen = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Transfer Learning: ResNeSt
backbone = ResNet50V2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)


# thêm regularization cho các lớp conv bên trong backbone
for layer in backbone.layers:
    if isinstance(layer, layers.Conv2D):
        layer.kernel_regularizer = regularizers.l2(1e-4)

backbone.trainable = False

# Xây dựng mô hình cuối
model = models.Sequential([
    backbone,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

# Hàm learning rate schedule (Cosine decay)
initial_lr = 1e-3
def cosine_decay(epoch):
    return initial_lr * 0.5 * (1 + math.cos(epoch / 50 * math.pi))

lr_scheduler = LearningRateScheduler(cosine_decay)

# Compile mô hình (Stage 1)
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=initial_lr),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_stage1 = [
    EarlyStopping(patience=6, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.3, patience=3, min_lr=1e-6),
    ModelCheckpoint("screw_best_ResNet_stage1.h5", save_best_only=True),
    lr_scheduler
]

# Stage 1: Feature extraction
print("\n STAGE 1: FEATURE EXTRACTION")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks_stage1
)

# Stage 2: Fine-tuning
print("\n STAGE 2: FINE-TUNING")
backbone.trainable = True
fine_tune_at = len(backbone.layers) - 60
for layer in backbone.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5,weight_decay=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_stage2 = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-7),
    ModelCheckpoint("screw_best_ResNet_stage2.h5", save_best_only=True)
]

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks_stage2
)

# Đánh giá mô hình
print("\n MODEL EVALUATION")
test_loss, test_acc = model.evaluate(test_gen)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

model.save("new_screw_resnest_finetuned.h5")
print("\nModel saved as 'new_screw_resnest_finetuned.h5'")

# Visualization: Accuracy & Loss trên một biểu đồ
acc      = history1.history['accuracy']      + history2.history['accuracy']
val_acc  = history1.history['val_accuracy']  + history2.history['val_accuracy']
loss     = history1.history['loss']          + history2.history['loss']
val_loss = history1.history['val_loss']      + history2.history['val_loss']
epochs   = range(1, len(acc) + 1)

plt.figure(figsize=(10,6))
plt.plot(epochs,     acc,     label='Train Accuracy',        color='blue')
plt.plot(epochs,     val_acc, label='Validation Accuracy',   color='orange')
plt.plot(epochs,     loss,    label='Train Loss',            color='green', linestyle='--')
plt.plot(epochs,     val_loss,label='Validation Loss',      color='red',   linestyle='--')
plt.title('Training Progress (Accuracy & Loss)')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Confusion Matrix & Classification Report
y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes
labels = list(test_gen.class_indices.keys())

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print("\n CLASSIFICATION REPORT")
print(classification_report(y_true, y_pred_classes, target_names=labels))

print("\n SUMMARY METRICS")
print(f"Train Accuracy (last epoch): {acc[-1]:.4f}")
print(f"Val Accuracy   (last epoch): {val_acc[-1]:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
