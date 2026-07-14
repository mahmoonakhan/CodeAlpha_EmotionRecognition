# CodeAlpha ML Internship - Emotion Recognition from Speech

## Task 2: Emotion Recognition from Speech

**Objective**: Recognize human emotions (happy, angry, sad) from speech audio using deep learning and speech signal processing techniques.

---

## Approach

- **Feature Extraction**: MFCCs (Mel-Frequency Cepstral Coefficients)
- **Deep Learning Model**: CNN + LSTM hybrid architecture
  - CNN layers for local spectral feature extraction
  - LSTM layers for temporal pattern learning
- **Dataset**: RAVDESS Emotional Speech Audio (Kaggle)

---

## Dataset

| Property | Details |
|----------|---------|
| Source | RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) |
| Total Actors | 24 (12 male, 12 female) |
| Emotions Used | happy, sad, angry |
| Audio Format | 16-bit, 48kHz WAV |
| Features Extracted | 40 MFCC coefficients |

---

## Model Architecture
Input: (174, 40) → MFCC spectrogram
↓
Conv1D(64) → BatchNorm → MaxPool → Dropout(0.3)
↓
Conv1D(128) → BatchNorm → MaxPool → Dropout(0.3)
↓
LSTM(128, return_sequences=True) → Dropout(0.3)
↓
LSTM(64) → Dropout(0.3)
↓
Dense(64) → BatchNorm → Dropout(0.3)
↓
Dense(3) → Softmax


---

## Results

| Metric | Score |
|--------|-------|
| **Accuracy** | **90.48%** |
| **Precision** | **90.64%** |
| **Recall** | **90.48%** |
| **F1-Score** | **90.52%** |

### Confusion Matrix

| Actual \ Predicted | happy | sad | angry |
|-------------------|-------|-----|-------|
| **happy** | 69 | 7 | 1 |
| **sad** | 5 | 70 | 2 |
| **angry** | 3 | 4 | 70 |

### Per-Emotion Performance

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| happy | 89.6% | 89.6% | 89.6% |
| sad | 86.4% | 90.9% | 88.6% |
| angry | 95.9% | 90.9% | 93.3% |

---

## Key Features

- ✅ MFCC feature extraction with Librosa
- ✅ CNN + LSTM deep learning architecture
- ✅ 3-class emotion classification (happy, sad, angry)
- ✅ Early stopping and learning rate reduction callbacks
- ✅ Comprehensive evaluation metrics and visualizations

---

## Technologies

- Python 3.12
- TensorFlow / Keras
- Librosa (audio processing)
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

---

## Usage
from tensorflow.keras.models import load_model
import joblib
import librosa
import numpy as np

# Load model and encoder
model = load_model('emotion_recognition_model.h5')
le = joblib.load('emotion_label_encoder.pkl')

# Predict emotion from audio
def predict_emotion(audio_path):
    audio, sr = librosa.load(audio_path, duration=3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.pad(mfcc, ((0,0), (0, 174-mfcc.shape[1])), mode='constant')
    mfcc = mfcc.T.reshape(1, 174, 40)
    
    pred = model.predict(mfcc)
    emotion = le.inverse_transform([np.argmax(pred)])[0]
    return emotion

# Example
emotion = predict_emotion('sample_audio.wav')
print(f"Predicted emotion: {emotion}")

# Author
Mahmoona Khan
CodeAlpha ML Intern
LinkedIn: https://www.linkedin.com/in/dataanalyst-mahmoona/


## Acknowledgments
Dataset: RAVDESS Emotional Speech Audio
Internship: @CodeAlpha


## Installation

```bash
pip install -r requirements.txt

---

