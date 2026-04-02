## Waste Classification REST API 

A lightweight image classification service built with Flask + TensorFlow.  
The service loads a fine-tuned Keras Applications model and exposes a REST API for real-time prediction.

## What this service does
- Classifies an uploaded image into metal or plastic wastes
- Returns:
   Top-1 predicted class + probability
   Top-k probabilities for transparency

## Dataset 
This model was trained using the Waste Classification dataset from Kaggle:
 classes in the data `METAL`, `PLASTIC`

