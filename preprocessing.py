import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
RAW_DATA_PATH = r"C:\Users\vadla\Documents\Echo-Guard\data\raw\2nd_test\2nd_test"# Adjust based on your extracted folder name
OUTPUT_PATH = r"C:\Users\vadla\Documents\Echo-Guard\data\processed"
SAMPLE_RATE = 20000 # The NASA sensors recorded at 20kHz

# Ensure output directories exist
os.makedirs(f"{OUTPUT_PATH}/healthy", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/faulty", exist_ok=True)

def generate_spectrogram_image(file_path, save_path):
    """
    Reads a raw vibration file, computes the spectrogram, and saves it as an image.
    """
    try:
        # 1. Read the data (NASA format is tab-separated, no header)
        df = pd.read_csv(file_path, sep='\t', header=None)
        
        # 2. Extract Bearing 1 data (Column 0)
        signal = df[0].values
        
        # 3. Short-Time Fourier Transform (STFT)
        # We assume the signal is short, so we take the whole chunk
        D = librosa.stft(signal)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # 4. Save as Image (No axes, just the data pattern)
        plt.figure(figsize=(4, 4))
        librosa.display.specshow(S_db, sr=SAMPLE_RATE)
        plt.axis('off') # We don't want axis numbers in our training images
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close() # Close memory to prevent crashing
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_dataset():
    # Get all filenames sorted by date
    files = sorted(os.listdir(RAW_DATA_PATH))
    
    # --- STRATEGY ---
    # The NASA "2nd_test" ran from Feb 12 to Feb 19.
    # The first ~200 files are definitely healthy.
    # The last ~50 files are definitely broken.
    # We ignore the middle (transition phase) to avoid confusing the model for now.
    
    healthy_files = files[:300] 
    faulty_files = files[-100:] 
    
    print(f"Processing {len(healthy_files)} Healthy files...")
    for i, f in enumerate(healthy_files):
        source = os.path.join(RAW_DATA_PATH, f)
        target = os.path.join(OUTPUT_PATH, "healthy", f"{i}.png")
        generate_spectrogram_image(source, target)
        if i % 50 == 0: print(f"  Processed {i} healthy...")

    print(f"Processing {len(faulty_files)} Faulty files...")
    for i, f in enumerate(faulty_files):
        source = os.path.join(RAW_DATA_PATH, f)
        target = os.path.join(OUTPUT_PATH, "faulty", f"{i}.png")
        generate_spectrogram_image(source, target)
        if i % 10 == 0: print(f"  Processed {i} faulty...")

    print("Done! Check your data/processed folder.")

if __name__ == "__main__":
    process_dataset()