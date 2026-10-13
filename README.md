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

## Actual CPU Bayesian training lane

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
python -m unittest discover -s tests -v
python cpu_bayes.py --epochs 2 --train-size 512 --test-size 128
```

The CPU lane downloads actual labeled CIFAR-10 examples from the university's hosted dataset, trains a variational CNN with Gaussian posterior reparameterization and per-example KL regularization, saves/reloads the model, and produces posterior-sampling predictions. BatchNorm remains in evaluation mode during Monte Carlo sampling while dropout and Bayesian weights remain stochastic. Data and parameters are real; this does not use mock inference. The hosted preview exports may re-encode images, so the measured subset accuracy is not a byte-identical canonical CIFAR benchmark. Sources: https://huggingface.co/datasets/uoft-cs/cifar10 and https://www.cs.toronto.edu/~kriz/cifar.html .

This implementation preserves the original convolutional channel layout but uses posterior reparameterization instead of TensorFlow Probability DenseFlipout; it is an explicitly separate small CPU experiment. It does not establish full-dataset performance or calibrated uncertainty. The original TensorFlow notebook is retained with a compatible legacy-Keras profile, KL scaling and corrected MC sampling that does not update BatchNorm. That TensorFlow profile was syntax-checked, not executed in this constrained runtime. Use requirements-tensorflow.txt plus `pip install tf-keras==2.16.0 --no-deps` in a separate environment to run it.
