## Waste Classification REST API 

- A lightweight image classification service built with Flask + TensorFlow.
- Transfer learning were applied using ResNet50, MobileNetV2 and InceptionV3 and the best model chosen.
- The service loads a fine-tuned Keras Applications model and exposes a REST API for real-time prediction.

#### Feature Extraction Phase
- Base model layers were frozen
- Only the custom classification head was trained
#### Fine-Tuning Phase
- Top layers of the base model were unfrozen
- Trained with a lower learning rate
- Fine-tuning improved validation accuracy for all models.
- ResNet50 achieved the best performance, therefore, ResNet50 was selected for deployment 

## What this service does
- Classifies an uploaded image into metal or plastic wastes
- Returns:
   Top-1 predicted class + probability
   Top-k probabilities for transparency

## Dataset 
This model was trained using the Waste Classification dataset from Kaggle:
 classes in the data `METAL`, `PLASTIC`

