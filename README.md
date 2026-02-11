# Improving MLP-Mixer Performance on Small Image Datasets

This repository contains the PyTorch implementation of a **Hybrid MLP-Mixer** architecture designed to bridge the gap between MLPs and CNNs on small datasets.

## 📌 Project Overview
[cite_start]The standard **MLP-Mixer** architecture relies entirely on matrix multiplications, lacking the built-in spatial inductive bias (locality) found in CNNs[cite: 8, 9, 66]. [cite_start]While it achieves state-of-the-art results on massive datasets (e.g., JFT-300M), it struggles to generalize on smaller benchmarks like CIFAR-10 and CIFAR-100, often leading to overfitting [cite: 10, 71-73].

[cite_start]**Our Solution:** We introduce a **Hybrid Convolutional Stem**—a lightweight block inserted immediately after patch embedding[cite: 89, 90]. [cite_start]This block uses a $3\times3$ Depthwise Convolution followed by a $1\times1$ Pointwise Convolution to "smooth" patch representations and inject local spatial awareness before global mixing begins [cite: 91-93].

## 🚀 Key Features
* **Hybrid Architecture:** Combines the efficiency of MLPs with the local feature extraction of CNNs.
* [cite_start]**Minimal Overhead:** The added stem uses Depthwise Separable Convolutions, adding only ~17.5k parameters (vs ~147k for a standard convolution) [cite: 106-109, 640-641].
* [cite_start]**Plug-and-Play:** The modification preserves the token count and dimensionality, making it compatible with existing Mixer blocks [cite: 127-130].

## 📊 Results
We evaluated the model on **CIFAR-10** and **CIFAR-100**. [cite_start]The Hybrid Mixer demonstrates faster convergence and superior generalization compared to the baseline MLP-Mixer [cite: 151, 171-200].

| Dataset | Baseline Accuracy | Hybrid (Ours) Accuracy | Improvement |
| :--- | :--- | :--- | :--- |
| **CIFAR-10** | 76.71% | **82.27%** | **+5.56%** |
| **CIFAR-100** | 48.20% | **55.03%** | **+6.83%** |

