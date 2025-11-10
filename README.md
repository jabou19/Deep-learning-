# 🧠 From Scratch to Insights: Feedforward Neural Networks in NumPy

**Training, Optimization, and Experiment Tracking with Weights & Biases (WandB)**

---

## 📘 Overview

This project implements a **fully-connected Feedforward Neural Network (FFNN)** **from scratch** using **NumPy**,  
without relying on high-level deep learning libraries such as TensorFlow or PyTorch.

The model supports **forward and backward propagation**, **gradient-based optimization**, and **experiment tracking** via **Weights & Biases (WandB)**.

The primary goal is to help intermediate learners **understand the mathematical foundations** of deep learning  
and how modern frameworks like PyTorch perform these steps under the hood.

---

## 🎯 Objectives

- Implement a complete FFNN using only NumPy
- Support configurable architecture and activation functions
- Implement **forward**, **backward**, and **parameter update** steps manually
- Train and evaluate on small datasets (e.g., Fashion-MNIST, CIFAR-10)
- Visualize and track experiments using **WandB**

---

## 🧩 Project Structure

```
DeepL_Project/
│
├── ffnn.py # Core network implementation (forward + backward + train)
├── utils.py # (optional) Helper functions for activations, metrics
├── train.py # Training & evaluation script
├── Test.py # Simple test / toy dataset experiments
├── data/ # (optional) Dataset loaders or local data
├── requirements.txt # Dependencies
├── wandb/ # Local WandB run logs
└── README.md # Project documentation
```
