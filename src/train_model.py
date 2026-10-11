import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.model import build_echo_guard_cnn

# --- CONFIGURATION ---
DATA_DIR = r"../data/processed"
IMG_HEIGHT = 256  # Resize images to this for the model
IMG_WIDTH = 256
BATCH_SIZE = 32

def train_model():
    print("Loading images from disk...")
    
    # 1. Load Data using Keras utility
    # This automatically splits into Train (80%) and Validation (20%)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    # 2. Performance Tuning (Prefetching data to RAM)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 3. Build the CNN Architecture
    print("Building the Neural Network using modular EchoGuard architecture...")
    model = build_echo_guard_cnn(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        dropout_rate=0.5,
        learning_rate=1e-3,
        compile_model=True
    )
    model.summary()

    # 4. Setup Callbacks
    model_save_path = "echo_guard_model.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    # 5. Train the Model
    print("Starting Training...")
    epochs = 15
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # 6. Save Final Model State
    model.save(model_save_path)
    print(f"Model saved successfully as '{model_save_path}'!")

    # 6. Visualize Results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), acc, label='Training Accuracy')
    plt.plot(range(epochs), val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), loss, label='Training Loss')
    plt.plot(range(epochs), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

if __name__ == "__main__":
    train_model()