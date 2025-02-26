import pickle as pkl
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import keras
from typing import Dict, Union


class MnistClassifierInterface(ABC):
    """
    Abstract base class for MNIST classifiers.
    """
    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the classifier with the given training data.

        Args:
            x_train (np.ndarray): Training images as a NumPy array.
            y_train (np.ndarray): Training labels as a NumPy array.
        """
        pass

    @abstractmethod
    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Predict the class of a given image.

        Args:
            image (Image.Image): Input image in PIL format.

        Returns:
            np.ndarray: Predicted class label as a NumPy array.
        """
        pass


class RandomForest(MnistClassifierInterface):
    def __init__(self, model_kwargs: Dict):
        """
        Initialize the RandomForest classifier by loading a pre-trained model.

        Args:
            model_kwargs (dict): Hyperparameters for the RandomForestClassifier.
        """
        self.model_kwargs = model_kwargs
        self.model: RandomForestClassifier = pkl.load(open("./models/random_forest.model", 'rb'))

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the RandomForest model using the provided dataset.
        """
        x_train = x_train.reshape(x_train.shape[0], -1) / 255.0  # Flatten and normalize images
        self.model = RandomForestClassifier(**self.model_kwargs)
        self.model.fit(x_train, y_train)

    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Predict the label of an input image using the trained RandomForest model.
        """
        image = image.convert('L')  # Convert to grayscale
        image_array = np.array(image).reshape(1, -1) / 255.0  # Flatten and normalize
        return self.model.predict(image_array)


class FeedForwardNeuralNetwork(MnistClassifierInterface):
    def __init__(self):
        """
        Initialize the FeedForward Neural Network classifier by loading a pre-trained model.
        """
        self.model: Sequential = pkl.load(open("./models/feed_forward_nn.model", 'rb'))

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the FeedForward neural network.
        """
        x_train = x_train / 255.0  # Normalize pixel values

        self.model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer=Adam(),
                           loss=SparseCategoricalCrossentropy(),
                           metrics=[SparseCategoricalAccuracy()])
        
        self.model.fit(x_train, y_train, epochs=50)

    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Predict the label of an input image using the trained FeedForward neural network.
        """
        image = image.convert('L')  # Convert to grayscale
        image_array = np.array(image).reshape(1, 28, 28) / 255.0  # Normalize and reshape
        prediction = self.model.predict(image_array)
        return np.argmax(prediction, axis=1)


class ConvolutionalNeuralNetwork(MnistClassifierInterface):
    def __init__(self):
        """
        Initialize the Convolutional Neural Network classifier by loading a pre-trained model.
        """
        self.model: Sequential = pkl.load(open("./models/conv_nn.model", 'rb'))

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the Convolutional Neural Network model.
        """
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) / 255.0  # Reshape and normalize
        y_train = keras.utils.to_categorical(y_train)  # One-hot encode labels

        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(100, activation='relu', kernel_initializer='he_uniform'),
            Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        
        self.model.fit(x_train, y_train, epochs=5, batch_size=500)

    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Predict the label of an input image using the trained Convolutional Neural Network.
        """
        image = image.convert('L').resize((28, 28))  # Convert to grayscale and resize
        image_array = np.array(image).reshape(1, 28, 28, 1) / 255.0  # Normalize and reshape
        prediction = self.model.predict(image_array)
        return np.argmax(prediction, axis=1)


class MnistClassifier:
    def __init__(self, algorithm: str, model_kwargs: Dict = {}):
        """
        Initialize the classifier based on the selected algorithm.

        Args:
            algorithm (str): The algorithm type ('rf' for RandomForest, 'nn' for FeedForward NN, 'cnn' for Convolutional NN).
            model_kwargs (dict): Additional parameters for model initialization.
        """
        if algorithm == "rf":
            self.classifier: MnistClassifierInterface = RandomForest(model_kwargs)
        elif algorithm == "nn":
            self.classifier = FeedForwardNeuralNetwork()
        elif algorithm == "cnn":
            self.classifier = ConvolutionalNeuralNetwork()
        else:
            raise ValueError("Invalid algorithm. Choose from 'rf', 'nn', or 'cnn'.")

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the selected classifier with the provided dataset.
        """
        self.classifier.train(x_train, y_train)

    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Predict the label of an input image using the selected classifier.
        """
        return self.classifier.predict(image)