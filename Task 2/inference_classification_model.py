import argparse
import numpy as np
import pickle as pkl
from tensorflow.keras.preprocessing import image
import os

def load_and_predict(model_path, img_path, img_size=(224, 224)):
    """
    Load a trained model and predict the class of an image.

    Args:
        model_path (str): Path to the trained model.
        img_path (str): Path to the input image.
        img_size (tuple): Target image size.

    Returns:
        str: Predicted class label.
    """
    model = pkl.load(open(model_path, 'rb'))
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return predicted_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on an image using a trained model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model file")
    parser.add_argument("--img_path", type=str, required=True, help="Path to input image")

    args = parser.parse_args()
    result = load_and_predict(args.model_path, args.img_path)
    print(f"Predicted class: {result}")