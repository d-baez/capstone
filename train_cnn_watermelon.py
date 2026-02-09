# train_cnn_watermelon.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import logging
import sys
from datetime import datetime
import json

# Setup logging
log_filename = f"logs/watermelon_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data_watermelon")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
MODEL_OUT = "watermelon_cnn.h5"

logger.info("=" * 60)
logger.info("WATERMELON RIPENESS DETECTION - TRAINING")
logger.info("=" * 60)
logger.info(f"Data directory: {DATA_DIR}")
logger.info(f"Image size: {IMG_SIZE}")
logger.info(f"Batch size: {BATCH_SIZE}")
logger.info(f"Epochs: {EPOCHS}")

train_dir = DATA_DIR / "train"
val_dir = DATA_DIR / "val"

logger.info("\nLoading datasets...")
train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

val_ds = keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

# Log class names
class_names = train_ds.class_names
logger.info(f"Classes: {class_names}")

# Performance tweaks
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

num_classes = 4

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.15),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
])


def make_watermelon_cnn(input_shape=(224, 224, 3), num_classes=4):
    inputs = keras.Input(shape=input_shape)

    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model


logger.info("\nBuilding model...")
model = make_watermelon_cnn(input_shape=IMG_SIZE + (3,), num_classes=num_classes)

# Log model summary to file
with open(log_filename.replace('.log', '_model_summary.txt'), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
model.summary()

total_params = model.count_params()
logger.info(f"Total parameters: {total_params:,}")

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


# Custom callback for logging
class LoggingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logger.info(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"loss: {logs['loss']:.4f} - "
            f"acc: {logs['accuracy']:.4f} - "
            f"val_loss: {logs['val_loss']:.4f} - "
            f"val_acc: {logs['val_accuracy']:.4f}"
        )


callbacks = [
    LoggingCallback(),
    keras.callbacks.ModelCheckpoint(
        MODEL_OUT, save_best_only=True, monitor="val_accuracy", mode="max"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    ),
]

logger.info("\nStarting training...")
logger.info("=" * 60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1  # Keep Keras progress bar
)

logger.info("=" * 60)
logger.info(f"Training complete! Saved best model to {MODEL_OUT}")

# Save training history
history_file = "watermelon_training_history.json"
history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(history_file, "w") as f:
    json.dump(history_dict, f, indent=2)
logger.info(f"Saved training history to {history_file}")

# Log final metrics
best_val_acc = max(history.history['val_accuracy'])
best_val_loss = min(history.history['val_loss'])
logger.info(f"\nBest validation accuracy: {best_val_acc:.4f}")
logger.info(f"Best validation loss: {best_val_loss:.4f}")

logger.info(f"\nLog saved to: {log_filename}")
