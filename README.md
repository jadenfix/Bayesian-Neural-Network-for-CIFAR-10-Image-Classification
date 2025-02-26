# Bayesian CNN for CIFAR-10 Image Classification 🧠📊

This project implements a **Simplified Bayesian Convolutional Neural Network (CNN)** using **TensorFlow** and **TensorFlow Probability** to classify images from the **CIFAR-10 dataset**. By incorporating Bayesian inference, the model estimates prediction **uncertainty**, making it more robust in real-world applications.

## 🔍 Project Overview
Traditional deep learning models often provide **point estimates** for classification but lack uncertainty quantification. Bayesian CNNs, on the other hand:
- Capture **predictive uncertainty** through **Monte Carlo Dropout** and **Bayesian layers**.
- Provide **better generalization** compared to deterministic CNNs.
- Improve **robustness** in noisy environments.

This repository demonstrates:
- A **Bayesian CNN** model with **`DenseFlipout`** layers for uncertainty estimation.
- **Monte Carlo sampling** to evaluate **confidence intervals** of predictions.
- Training and evaluation on **CIFAR-10**, a benchmark image classification dataset.

## 🚀 Features
✅ **Data Augmentation**: Improves model generalization using `ImageDataGenerator`.  
✅ **Bayesian CNN Architecture**: Uses `DenseFlipout` layers from **TensorFlow Probability**.  
✅ **Monte Carlo Sampling**: Estimates predictive uncertainty over multiple iterations.  
✅ **Exponential Decay Learning Rate**: Optimizes model training over epochs.  
✅ **Pretrained Models for Benchmarking**: Includes a **MobileNetV2** and **ResNet50** implementation for comparison.

## 📂 Code Structure
- [`ComputerVision.ipynb`](https://github.com/jadenfix/Bayesian-Neural-Network-for-CIFAR-10-Image-Classification/blob/main/ComputerVision.ipynb) - Main Jupyter Notebook implementing the Bayesian CNN.
- `data/` - Folder containing the CIFAR-10 dataset.
- `models/` - Pretrained models and saved checkpoints.

## 📊 Model Architecture
```text
📸 Input: 32x32 RGB Image (CIFAR-10)
🔍 Conv2D (32 filters, ReLU) → BatchNorm → MaxPooling
🔍 Conv2D (64 filters, ReLU) → BatchNorm → MaxPooling
🔍 Conv2D (128 filters, ReLU) → BatchNorm → MaxPooling
🧠 Flatten → DenseFlipout (128 units, ReLU) → Dropout (40%)
🎯 DenseFlipout (10 classes, Softmax)
