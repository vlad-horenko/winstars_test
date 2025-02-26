import numpy as np
import pickle as pkl
from transformers import pipeline
from typing import Union
from PIL import Image

class ClassificationPipeline:
    """
    A pipeline for zero-shot text classification and image classification.
    Uses a transformer-based model for text classification and a pre-trained
    machine learning model for image classification.
    """
    
    def __init__(self) -> None:
        """
        Initializes the classification pipeline by loading the models.
        """
        self.ner_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.classification_model = pkl.load(open("./model/animal_class.model", 'rb'))

    def predict(self, text_input: str, image_input: Image.Image) -> int:
        """
        Predicts whether the text and image represent the same animal class.

        Args:
            text_input (str): The input text describing an animal.
            image_input (Image.Image): The input image of an animal.

        Returns:
            int: 1 if the text and image represent the same class, otherwise 0.
        """
        animals = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
        result = self.ner_model(text_input, animals)
        
        try:
            user_class = [label for label, score in zip(result["labels"], result["scores"]) if score > 0.5][0]
        except IndexError:
            return 0  # No confident classification found in text

        # Preprocess image: resize to (224, 224) and convert to RGB
        image = image_input.resize((224, 224))
        image = image.convert("RGB")
        
        # Normalize and reshape the image for the classification model
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        
        # Predict the class of the image
        image_class = self.classification_model.predict(image_array)
        image_class = np.argmax(image_class, axis=1)  # Get the class index

        classification_model_classes = {
            0: 'butterfly', 1: 'cat', 2: 'chicken', 3: 'cow', 4: 'dog',
            5: 'elephant', 6: 'horse', 7: 'sheep', 8: 'spider', 9: 'squirrel'
        }

        # Compare predicted image class with text classification result
        return 1 if user_class == classification_model_classes[image_class[0]] else 0
