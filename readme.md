# Task 4-3 — Waste Classification REST API (METAL vs PLASTIC)

A lightweight image classification service built with **Flask + TensorFlow**.  
The service loads a fine-tuned **Keras Applications** model (selected as the best model from Task 4-2) and exposes a REST API for real-time prediction.

## What this service does
- Classifies an uploaded image into:
  - **METAL**
  - **PLASTIC**
- Returns:
  - Top-1 predicted class + probability
  - Top-k probabilities for transparency

## Dataset (training source)
This model was trained using the **Waste Classification** dataset from Kaggle:
- Classes used: `METAL`, `PLASTIC`

> Note: The dataset is not included in this repository.

## Project structure