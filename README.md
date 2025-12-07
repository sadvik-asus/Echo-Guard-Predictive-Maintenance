# Echo-Guard: Industrial Predictive Maintenance System 🔊

**Echo-Guard** is an end-to-end Deep Learning application designed to predict machinery failure before it happens. It processes raw vibration sensor data from the NASA Bearing Dataset, converts signals into Mel-Spectrograms, and uses a Convolutional Neural Network (CNN) to classify equipment health in real-time.

## 🚀 Key Features
* **Signal Processing:** Automated pipeline to convert Time-Domain vibration data to Frequency-Domain Spectrograms using `Librosa`.
* **Deep Learning:** Custom 2D-CNN architecture built in `TensorFlow/Keras` achieving >95% accuracy.
* **Real-Time Dashboard:** Interactive User Interface built with `Streamlit` for live sensor monitoring.
* **Fault Detection:** Distinguishes between "Healthy" operation and "Critical" bearing degradation.

## 🛠 Tech Stack
* **Python 3.9**
* **TensorFlow/Keras** (CNN Implementation)
* **Librosa** (Audio Feature Extraction)
* **Streamlit** (Frontend)
* **Pandas/NumPy** (Data Engineering)

## 📸 How It Works
1.  **Input:** System accepts raw NASA sensor data (or .wav files).
2.  **Preprocessing:** Applies Short-Time Fourier Transform (STFT) to generate a spectrogram.
3.  **Inference:** The CNN analyzes the visual pattern of the spectrogram.
4.  **Output:** Returns a confidence score and a Go/No-Go maintenance alert.

## 💻 How to Run
```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Run the Dashboard
streamlit run app.py
