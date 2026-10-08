import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
from collections import Counter

# ========== 1. ĐƯỜNG DẪN & CẤU HÌNH ==========
base_dir = r"D:\FPTU\Semester 5\DPL302m\datasets\Screw Datasets\screw_all"
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

img_size = (128, 128)
batch_size = 32
epochs = 10
learning_rate = 1e-4

# ========== 2. DATA AUGMENTATION & TẠO DATA GENERATOR ==========
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
)

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

valid_gen = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

print("Class mapping:", train_gen.class_indices)

# ========== 3. CLASS WEIGHTS ==========
counter = Counter(train_gen.classes)
total = sum(counter.values())
class_weights = {i: total / (len(counter) * n) for i, n in counter.items()}
print("Class weights:", class_weights)

# ========== 4. MÔ HÌNH TRANSFER LEARNING ==========
from tensorflow.keras.applications import MobileNetV2

base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base

inputs = tf.keras.Input(shape=(128,128,3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(train_gen.num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ========== 5. CALLBACKS ==========
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint(os.path.join(base_dir, 'best_cnn_model.h5'),
                                       monitor='val_accuracy',
                                       save_best_only=True,
                                       verbose=1)
]

# ========== 6. HUẤN LUYỆN (BASE LAYERS) ==========
print("\n--- Training CNN Model (Base Layers Frozen) ---")
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=epochs,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# ========== 7. ĐÁNH GIÁ SAU GIAI ĐOẠN 1 ==========
test_loss, test_acc = model.evaluate(test_gen)
print(f"\n[Test] Accuracy sau giai đoạn 1: {test_acc*100:.2f}%")

# ========== 8. FINE-TUNING ==========
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Freeze phần lớn, chỉ fine-tune phần cuối
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- Fine-tuning Last Layers ---")
fine_tune_history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=10,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# ========== 9. ĐÁNH GIÁ SAU FINE-TUNING ==========
ft_loss, ft_acc = model.evaluate(test_gen)
print(f"\n[Test] Accuracy sau fine-tuning: {ft_acc*100:.2f}%")

# ========== 10. VẼ BIỂU ĐỒ ==========
plt.figure(figsize=(12,5))

# Gộp kết quả 2 giai đoạn
acc = history.history['accuracy'] + fine_tune_history.history['accuracy']
val_acc = history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
loss = history.history['loss'] + fine_tune_history.history['loss']
val_loss = history.history['val_loss'] + fine_tune_history.history['val_loss']

# Accuracy
plt.subplot(1,2,1)
plt.plot(acc, label='Train acc')
plt.plot(val_acc, label='Val acc')
plt.axvline(len(history.history['accuracy'])-1, color='gray', linestyle='--', label='Start Fine-tuning')
plt.title('Training & Fine-tuning Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(loss, label='Train loss')
plt.plot(val_loss, label='Val loss')
plt.axvline(len(history.history['loss'])-1, color='gray', linestyle='--')
plt.title('Training & Fine-tuning Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ========== 11. LƯU MÔ HÌNH ==========
model.save(os.path.join(base_dir, 'mobilenetv2_finetuned.h5'))
print("\n Model đã được lưu thành công tại:", os.path.join(base_dir, 'mobilenetv2_finetuned.h5'))
