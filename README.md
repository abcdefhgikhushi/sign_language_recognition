# Sign Language Recognition System
## 📋 Overview
This project implements a real-time American Sign Language (ASL) alphabet recognition system using Convolutional Neural Networks (CNN) and computer vision techniques. The system can recognize ASL hand gestures for 29 classes through a webcam feed with 98% accuracy.

## ✨ Features
• **Real-time Recognition:** Live ASL alphabet detection via webcam

• **High Accuracy:** Achieved 98% test accuracy on the validation dataset

• **Data Collection Tool:** Custom data collection script for gathering training samples

• **Robust Preprocessing:** Advanced image preprocessing pipeline using MediaPipe and OpenCV

• **Easy to Use:** Simple interface for both training and prediction

• **Extensible:** Modular code structure for easy modifications and improvements

## 🎯 Demo

[🎬 Watch Demo Video](https://github.com/ancdefhgikhushi/sign_language_recognition/tree/main/demo/demo_video.mp4)

## 🏗️ Architecture
The project uses a custom CNN architecture optimized for hand gesture recognition:

• **Input Layer:** 224x224x3 RGB images

• **Convolutional Layers:** Multiple Conv2D layers with ReLU activation

• **Pooling Layers:** MaxPooling for dimensionality reduction

• **Dropout Layers:** Regularization to prevent overfitting

• **Dense Layers:** Fully connected layers for classification

• **Output Layer:** 29 classes (A-Z, DELETE, NOTHING, SPACE) with softmax activation

## 📁 Project Structure

```SignLanguageRecognition/
├── 📁 data/
│   ├── 📁 raw_images/                    # Original ASL alphabet dataset
│   └── 📁 processed_data/                # Preprocessed .npy files
│       ├── 📄 X_train_batch_0.npy
│       ├── 📄 y_train.npy
│       └── 📄 label_encoder.pkl
│
├── 📁 models/
│   └── 📁 saved_models/
│       └── 📄 best_model.h5              # Trained CNN model
│
├── 📁 src/
│   ├── 🐍 data_collection.py            # Real-time data collection (500 frames/class)
│   ├── 🐍 data_preprocessing.py         # Image preprocessing pipeline
│   ├── 🐍 model_training.py             # CNN model training script
│   ├── 🐍 model_testing.py              # Model evaluation and testing
│   └── 🐍 realtime_detection.py         # Real-time ASL recognition app
│
├── 📁 utils/
│   ├── 🐍 config.py                     # Configuration and constants
│   └── 🐍 helper_functions.py           # Utility functions
│
├── 📋 requirements.txt                   # Python dependencies
├── 📖 README.md                          # Project documentation
└── 📄 .gitignore                         # Git ignore file
```

## 🚀 Getting Started

### Prerequisites

• Python 3.8 or higher

• Webcam for real-time detection

• GPU (optional, for faster training)

### Installation

#### 1. Clone the repository

#### 2. Create a virtual environment

#### 3. Install dependencies

## 📊 Dataset Setup

The project uses the ASL Alphabet dataset for training. You can either:

1. **Download the pre-processed dataset** (recommended for quick start)

2. **Use the custom data collection tool:**

   ```bash
   python src/data_collection.py
   ```
   This will collect 500 frames for each ASL alphabet class through your webcam.

## 🔧 Usage

### 1. Preprocess the data

```bash
python src/data_preprocessing.py
```
### 2. Train the CNN model

```bash
python src/model_training.py
```

### 3. Test the model

```bash
python src/model_testing.py
```

### 4. Run the real-time ASL recognition system

```bash
python src/realtime_detection.py
```

## 📈 Results

• **Test Accuracy:** 98%

• **Training Time:** Depends on hardware and number of frames

### Performance Metrics

| Metric | Value |
|--------|-------|
| Test Accuracy | 98.0% |
| Precision | 97.8% |
| Recall | 97.9% |
| F1-Score | 97.8% |

## 🛠️ Technical Details

### Dependencies

• **TensorFlow 2.16.1:** Deep learning framework

• **OpenCV 4.8.1:** Computer vision operations

• **MediaPipe 0.10.21:** Hand landmark detection

• **NumPy 1.26.0:** Numerical computations

• **Matplotlib 3.8.0:** Data visualization

• **Scikit-learn 1.3.2:** Machine learning utilities

• **Pillow 10.1.0:** Image processing

## 🔮 Future Enhancements

 • Add support for ASL words and phrases
 
 • Implement gesture sequence recognition
 
 • Add mobile app version
 
 • Support for multiple hand detection
 
 • Integration with speech synthesis
 
 • Add more robust background handling










   


