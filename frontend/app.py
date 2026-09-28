import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import os
import gdown
# --- PAGE CONFIGURATION ---
# st.set_page_config(page_title="Echo-Guard Dashboard", page_icon="🔊", layout="wide")

# # --- LOAD MODEL ---
# # We use @st.cache_resource so we only load the heavy model once, not every time you click a button
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model(r"C:\Users\vadla\Documents\Echo-Guard\src\echo_guard_model.keras")

# model = load_model()
@st.cache_resource
def load_model_from_drive():
    # Replace with your actual file ID from Google Drive
    file_id = '1A_slcn6AxLLuo7UP3qrc7JOMfgdKK-xy'
    url = f'https://drive.google.com/file/d/1A_slcn6AxLLuo7UP3qrc7JOMfgdKK-xy/view?usp=sharing'
    output = 'echo_guard_model.keras'
    
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    
    return tf.keras.models.load_model(output)

# Use the function to load the model
model = load_model_from_drive()
# --- UTILITY FUNCTIONS ---
def create_spectrogram(audio_file):
    """
    Converts uploaded audio -> Spectrogram Image -> Returns image path
    """
    try:
        # Check if it looks like a standard audio file
        if audio_file.name.lower().endswith(('.wav', '.mp3', '.flac')):
             signal, sr = librosa.load(audio_file, sr=20000)
        
        # If no extension or .txt/.csv, treat as NASA format
        else:
             import pandas as pd
             # Reset file pointer just in case
             audio_file.seek(0)
             df = pd.read_csv(audio_file, sep='\t', header=None)
             signal = df[0].values
             sr = 20000
             
    except Exception as e:
        # If the first attempt failed, try the fallback (Force NASA format)
        import pandas as pd
        audio_file.seek(0)
        df = pd.read_csv(audio_file, sep='\t', header=None)
        signal = df[0].values
        sr = 20000

    # Create Spectrogram
    D = librosa.stft(signal)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Save to a temporary buffer
    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_db, sr=sr)
    plt.axis('off')
    
    temp_img_path = "temp_spec.png"
    plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return temp_img_path
# def create_spectrogram(audio_file):
#     """
#     Same logic as our preprocessing script.
#     Converts uploaded audio -> Spectrogram Image -> Returns image path
#     """
#     # Load audio (Librosa handles many formats, but NASA data is specific)
#     # For this demo, we assume the user uploads one of the NASA files or a standard .wav
#     try:
#         # If it's a NASA text file
#         if audio_file.name.endswith('.txt') or audio_file.name.endswith('.csv'):
#              df = pd.read_csv(audio_file, sep='\t', header=None)
#              signal = df[0].values
#              sr = 20000
#         # If it's a standard audio file (wav/mp3)
#         else:
#              signal, sr = librosa.load(audio_file, sr=20000)
#     except:
#         # Fallback for the NASA raw text format if librosa fails directly
#         import pandas as pd
#         df = pd.read_csv(audio_file, sep='\t', header=None)
#         signal = df[0].values
#         sr = 20000

#     # Create Spectrogram
#     D = librosa.stft(signal)
#     S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
#     # Save to a temporary buffer
#     plt.figure(figsize=(4, 4))
#     librosa.display.specshow(S_db, sr=sr)
#     plt.axis('off')
    
#     temp_img_path = "temp_spec.png"
#     plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0)
#     plt.close()
    
#     return temp_img_path

def predict_health(image_path):
    # 1. Load the image we just saved
    img = tf.keras.utils.load_img(image_path, target_size=(256, 256))
    
    # 2. Convert to Array
    img_array = tf.keras.utils.img_to_array(img)
    
    # 3. Create a Batch (The model expects a batch of images, not just one)
    # Shape becomes (1, 256, 256, 3)
    img_array = tf.expand_dims(img_array, 0) 
    
    # 4. Predict
    prediction = model.predict(img_array)
    score = prediction[0][0] # The output is a probability between 0 and 1
    return score

# --- DASHBOARD UI ---
st.title("🔊 Echo-Guard: Industrial Predictive Maintenance")
st.markdown("Upload a vibration sensor reading to detect potential machinery failure.")

col1, col2 = st.columns(2)

with col1:
    st.header("1. Sensor Data Input")
    # uploaded_file = st.file_uploader("Upload Sensor File (NASA format or .wav)", type=['txt', 'csv', 'wav'])
    uploaded_file = st.file_uploader("Upload Sensor File (NASA format or .wav)", accept_multiple_files=False)

    if uploaded_file is not None:
        st.write("Processing Signal...")
        # Generate the visual
        spec_path = create_spectrogram(uploaded_file)
        st.image(spec_path, caption="Generated Mel-Spectrogram", use_container_width=True)
        
        # Run Prediction
        confidence = predict_health(spec_path)
        
        # Interpretation (Sigmoid output: Closer to 0 is Healthy, Closer to 1 is Faulty)
        # Note: Depending on how your folders were sorted, 0 might be faulty and 1 healthy.
        # usually flow is alphabetical: 0=Faulty, 1=Healthy OR if you used my script exactly:
        # We need to verify standard Keras flow. Usually alphanumeric.
        # 'faulty' comes before 'healthy' alphabetically? No. 'f' comes before 'h'.
        # So likely 0=Faulty, 1=Healthy. 
        # Let's add a logic check in the UI to be safe or just interpret based on probability.
        
        # Let's assume High Probability (1.0) = Class 1 (Healthy) and Low (0.0) = Class 0 (Faulty)
        # We will display the raw score to be sure.
        
        # ACTUALLY: Let's trust the 'image_dataset_from_directory' defaults.
        # It assigns 0 to the first folder alphabetically ('faulty') and 1 to ('healthy').
        # So: Score < 0.5 = FAULTY. Score > 0.5 = HEALTHY.

with col2:
    st.header("2. AI Diagnosis")
    
    if uploaded_file is not None:
        st.metric(label="Model Confidence Score", value=f"{confidence:.4f}")
        
        if confidence > 0.5:
            st.success("✅ SYSTEM HEALTHY")
            st.write("The vibration patterns indicate normal operation. No maintenance required.")
        else:
            st.error("🚨 CRITICAL FAULT DETECTED")
            st.write("Abnormal vibration patterns detected. Inspect bearing components immediately.")

            st.warning("Recommended Action: Schedule downtime for maintenance.")
