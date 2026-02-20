# Task 4-3 — Waste Classification REST API 

A lightweight image classification service built with **Flask + TensorFlow**.  
The service loads a fine-tuned **Keras Applications** model and exposes a REST API for real-time prediction.

## What this service does
- Classifies an uploaded image into:
- Returns:
  - Top-1 predicted class + probability
  - Top-k probabilities for transparency

## Dataset (training source)
This model was trained using the **Waste Classification** dataset from Kaggle:
- Classes used: `METAL`, `PLASTIC`

