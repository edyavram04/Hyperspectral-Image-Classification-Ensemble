# Hyperspectral Image Classification Ensemble

## Overview
This repository contains the source code for a machine learning solution developed for a hyperspectral image classification competition. The objective is to accurately classify small hyperspectral patches into distinct thematic classes, evaluated using the Macro F1-Score metric.

The final architecture implements an ensemble approach, combining a LightGBM classifier with a Multi-Layer Perceptron (MLP) Deep Neural Network, and utilizes Pseudo-Labeling (semi-supervised learning) to optimize performance on the unlabelled test distribution.

## Dataset and Data Processing
The dataset consists of hyperspectral image patches extracted from a larger scene. 
* **Dimensions:** Each patch is 19x19 pixels with 48 spectral bands.
* **Core Constraint:** The class assignment of a given patch is strictly determined by its central pixel.
* **Preprocessing:** Classes 4 and 5 were excluded from the training distribution based on exploratory data analysis and competition parameters. Continuous features were scaled using a Yeo-Johnson PowerTransformer to stabilize variance, specifically optimizing the neural network convergence.

## Feature Engineering Pipeline
Given the high dimensionality of the hyperspectral data, a custom feature extraction pipeline was implemented, inherently prioritizing the central spatial location:
1. **Spectral Features:** Direct extraction of the 48 bands from the central pixel.
2. **Derivatives:** 1st and 2nd order discrete differences to capture the slope and concavity of the spectral signature.
3. **Contextual Statistics:** Mean and standard deviation computed over the immediate 3x3 neighborhood surrounding the center.
4. **Global Statistics:** Skewness, Kurtosis, and Area under the curve (using the trapezoidal rule) to quantify the signal distribution.
5. **Custom Ratios:** An engineered index (IR to Blue ratio) designed to emphasize specific material or vegetation signatures.

## Model Architecture and Training Strategy

### Stage 1: Initial Ensemble Training
* **LightGBM:** Trained on the engineered feature set using a 10-Fold Stratified Cross-Validation setup (`class_weight='balanced'`).
* **MLP:** A 3-hidden-layer deep network (512 -> 256 -> 128 neurons) incorporating Batch Normalization and Dropout regularization.
* **Ensemble Blending:** Initial predictions were generated utilizing a weighted average: `40% LightGBM + 60% MLP`.

### Stage 2: Pseudo-Labeling
To leverage the unlabelled test data and adapt to its specific distribution:
* Predictions from Stage 1 with a confidence probability of >= 95% were extracted.
* These high-confidence test samples were assigned their predicted labels (pseudo-labels) and merged with the original training set.
* Both LightGBM and MLP models were re-trained from scratch on this augmented dataset.
* The final submission was generated using the identical 40/60 weighted blend from the newly trained models.
