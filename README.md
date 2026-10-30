# Handwritten Digit Recognition Using Neural Networks (MATLAB)

## Overview

This project investigates the influence of neural network architecture and complexity on handwritten digit classification. Implemented in MATLAB, the study compares perceptron-based models with a multilayer neural network to examine classification performance, learning behaviour and the limitations of linear decision boundaries.

## Implemented Models

**1. Single Perceptron**
- Binary classification of manually constructed 5 × 5 images representing digits 0 and 1.
- Implementation of the perceptron learning rule.
- Evaluation across 10 independent training and testing trials.

**2. Single-Layer Perceptron (SLP)**
- Classification of handwritten digits 0–9 using the MNIST dataset.
- Training using sigmoid activation and weight updates.
- Evaluation using learning curves, confusion matrices and visualisations of learned weights.

**3. Multilayer Perceptron (MLP)**
- Feedforward neural network with a hidden layer.
- Implementation of the backpropagation learning algorithm.
- Investigation of how hidden-layer capacity affects classification performance.

## Key Findings

- The binary perceptron achieved an average test accuracy of 85% across 10 trials on the small, manually constructed dataset.
- Learned-weight visualisations demonstrated how the networks identify patterns associated with individual digits.
- Confusion matrices highlighted classification difficulties between visually similar digits.
- Comparing network architectures illustrated the role of nonlinear modelling in handwritten digit recognition.

## Technologies

MATLAB · Neural Networks · Machine Learning · Image Classification · Data Visualisation

## Running the Code

The implementations require MATLAB. The MNIST-based scripts additionally require a compatible `mnist.mat` dataset containing the training and testing arrays used in the original coursework.

Place the dataset where the scripts can access it, then run each experiment individually in MATLAB.

## Project Context

Developed as part of the Machine Learning Lab coursework at the University of Manchester, December 2025.
