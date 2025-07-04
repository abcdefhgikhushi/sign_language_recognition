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
