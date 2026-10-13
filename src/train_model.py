import os
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.model import build_echo_guard_cnn
from src.config import (
    PROCESSED_DATA_PATH,
    DEFAULT_MODEL_PATH,
    IMG_HEIGHT,
    IMG_WIDTH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    DROPOUT_RATE,
    VALIDATION_SPLIT,
    RANDOM_SEED,
)

def train_model(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, show_plot: bool = True):
    print("Loading images from disk...")
    
    # Verify dataset directory exists and has image data
    if not PROCESSED_DATA_PATH.exists():
        print(f"Error: Processed data path {PROCESSED_DATA_PATH} not found.")
        print("Please run 'python src/generate_sample_data.py' or 'python src/preprocessing.py' first.")
        return

    # 1. Load Data using Keras utility
    # This automatically splits into Train and Validation sets
    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(PROCESSED_DATA_PATH),
            validation_split=VALIDATION_SPLIT,
            subset="training",
            seed=RANDOM_SEED,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            str(PROCESSED_DATA_PATH),
            validation_split=VALIDATION_SPLIT,
            subset="validation",
            seed=RANDOM_SEED,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE
        )
    except ValueError as err:
        print(f"Error loading dataset: {err}")
        print("Ensure 'data/processed/healthy' and 'data/processed/faulty' contain spectrogram PNGs.")
        return

    # 2. Performance Tuning (Prefetching data to RAM)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 3. Build the CNN Architecture
    print("Building the Neural Network using modular EchoGuard architecture...")
    model = build_echo_guard_cnn(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
        compile_model=True
    )
    model.summary()

    # 4. Setup Callbacks
    model_save_path = str(DEFAULT_MODEL_PATH)
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
    epochs = EPOCHS
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # 6. Save Final Model State
    model.save(model_save_path)
    print(f"Model saved successfully as '{model_save_path}'!")

    # 6. Visualize and Save Training Artifacts
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs_trained = list(range(1, len(acc) + 1))

    # Export metrics JSON
    metrics_path = DEFAULT_MODEL_PATH.parent / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "epochs_completed": len(epochs_trained),
            "final_train_accuracy": float(acc[-1]) if acc else None,
            "final_val_accuracy": float(val_acc[-1]) if val_acc else None,
            "final_train_loss": float(loss[-1]) if loss else None,
            "final_val_loss": float(val_loss[-1]) if val_loss else None,
            "history": history.history
        }, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_trained, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs_trained, val_acc, label='Validation Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training & Validation Accuracy')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_trained, loss, label='Training Loss', marker='o')
    plt.plot(epochs_trained, val_loss, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training & Validation Loss')
    plt.grid(True, linestyle='--', alpha=0.5)

    history_plot_path = DEFAULT_MODEL_PATH.parent / "training_history.png"
    plt.savefig(history_plot_path, bbox_inches='tight')
    print(f"Training curves saved to {history_plot_path}")

    if show_plot:
        plt.show()
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EchoGuard CNN Training Pipeline")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size")
    parser.add_argument("--no-plot", action="store_true", help="Suppress GUI plot window")
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size, show_plot=not args.no_plot)