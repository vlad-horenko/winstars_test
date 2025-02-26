# Task 2

For this task, I used the public Animal-10 dataset hosted on the Kaggle platform. To get the animal's name from the text, I used a public model hosted on Huggingface. For image classification, I trained my own model using the tensorflow framework.
This repository contains scripts for training and running inference on an image classification model using TensorFlow and Keras.

## 📌 Requirements
Ensure you have the following dependencies installed before running the scripts:
```bash
pip install -r requirements/requirements.txt
```

## 🚀 Training the Model
To train the image classification model, run the following command:
```bash
python train_classification_model.py --data_dir dataset/raw-img --model_save_path model/trained_model.keras --epochs 15 --batch_size 32 --learning_rate 0.001
```
### Arguments:
- `--data_dir` – Path to the dataset directory
- `--model_save_path` – Path to save the trained model
- `--epochs` – Number of training epochs (default: 15)
- `--batch_size` – Batch size for training (default: 32)
- `--learning_rate` – Learning rate for the optimizer (default: 0.001)

The model will be trained and saved to the specified path.

## 🔍 Running Inference
To make predictions using the trained model, run:
```bash
python inference_classification_model.py --model_path model/trained_model.keras --img_path "da
taset/raw-img/butterfly/e030b20a20e90021d85a5854ee454296eb70e3c818b413449df6c87ca3ed_640.jpg"
```
### Arguments:
- `--model_path` – Path to the trained model file
- `--img_path` – Path to the image for classification

The script will output the predicted class of the input image.

## 📂 Project Structure
```
.
├── dataset/raw-img/                           # Dataset directory
├── model/animal_class.model                   # Trained model (saved after training)
├── requirements/requirements.txt              # Libraries to be installed and their versions
├── demo.ipynb                                 # Demo notebook
├── train_classification_model.py              # Script for training the model
├── inference_classification_model.py          # Script for making predictions
├── task2.py                                   # Script to compare user input and image
└── README.md                                  # Project documentation
```

## 📌 Author
[Vladyslav Horenko] – Data Scientist