import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

# ------------ CONFIG ------------
base_dir = r"D:\FPTU\Semester 5\DPL302m\datasets\Screw Datasets\screw_all_clean"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "valid")
test_dir = os.path.join(base_dir, "test")

image_size = 160
batch_size = 32
initial_lr = 3e-4
epochs = 30
use_focal_loss = False

# ------------ Data Generators ------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, 
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=(image_size, image_size),
    batch_size=batch_size, class_mode='categorical', shuffle=False
)
test_gen = val_datagen.flow_from_directory(
    test_dir, target_size=(image_size, image_size),
    batch_size=batch_size, class_mode='categorical', shuffle=False
)

class_names = list(train_gen.class_indices.keys())
num_classes = len(class_names)

print("Classes:", class_names)
print("Train samples:", Counter(train_gen.classes))


# ------------ Custom Patch & Encoder Layers ------------
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, projection_dim, max_patches=256):
        super().__init__()
        self.projection = layers.Dense(units=projection_dim)
        self.max_patches = max_patches

    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(1, self.max_patches, self.projection.units),
            initializer="random_normal",
            trainable=True
        )

    def call(self, patches):
        x = self.projection(patches)
        num_patches = tf.shape(x)[1]
        pos_emb = self.pos_emb[:, :num_patches, :]
        return x + pos_emb



# ------------ ViT Model ------------
def small_vit_model(
    image_size=image_size,
    patch_size=16,
    projection_dim=64,
    transformer_layers=4,
    num_heads=4,
    mlp_dim=128,
    dropout_rate=0.2,
    num_classes=num_classes
):
    inputs = layers.Input(shape=(image_size, image_size, 3))
    # Conv stem
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    # Patch creation & encoding
    patches = Patches(patch_size)(x)
    encoded_patches = PatchEncoder(projection_dim)(patches)


    # Transformer blocks
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(mlp_dim, activation="gelu")(x3)
        x3 = layers.Dropout(dropout_rate)(x3)
        x3 = layers.Dense(projection_dim)(x3)
        encoded_patches = layers.Add()([x2, x3])

    # Classification head
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.GlobalAveragePooling1D()(representation)
    representation = layers.Dense(128, activation="relu")(representation)
    representation = layers.Dropout(0.4)(representation)
    outputs = layers.Dense(num_classes, activation="softmax")(representation)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


# ------------ Focal Loss ------------
def categorical_focal_loss(alpha=0.25, gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_ent = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_ent, axis=1)
    return loss_fn


loss_fn = categorical_focal_loss() if use_focal_loss else "categorical_crossentropy"

# ------------ Compile Model ------------
model = small_vit_model()
opt = tf.keras.optimizers.Adam(learning_rate=initial_lr, clipnorm=1.0)
model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])
model.summary()

# ------------ Class Weight ------------
cw = compute_class_weight('balanced', classes=np.unique(train_gen.classes), y=train_gen.classes)
class_weight = dict(enumerate(cw))
print("Using class weights:", class_weight)


# ------------ Callbacks ------------
class ValPredInspector(Callback):
    def __init__(self, val_gen, n=32):
        super().__init__()
        self.val_gen = val_gen
        self.n = n

    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = next(self.val_gen)
        preds = self.model.predict(x_val[:self.n])
        avg_pred = np.mean(preds, axis=0)
        print(f"\n[Epoch {epoch+1}] Avg val pred dist (first {self.n}):", np.round(avg_pred, 3))
        print("True counts:", np.bincount(np.argmax(y_val[:self.n], axis=1), minlength=len(avg_pred)))


callbacks = [
    EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss"),
    ReduceLROnPlateau(factor=0.3, patience=3, monitor="val_loss"),
    ModelCheckpoint("best_vit_model.h5", save_best_only=True),
    ValPredInspector(val_gen)
]

# ------------ Training ------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=callbacks,
    class_weight=class_weight
)

# ------------ Evaluation ------------
y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print(classification_report(y_true, y_pred_classes, target_names=class_names))
