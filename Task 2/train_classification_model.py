import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report

def train_model(data_dir, model_save_path, epochs=15, batch_size=32, learning_rate=0.001):
    """
    Train an image classification model using InceptionV3 as a feature extractor.

    Args:
        data_dir (str): Path to the dataset.
        model_save_path (str): Path to save the trained model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
    """
    img_height, img_width = 224, 224

    datagen = ImageDataGenerator(
        rescale=1/255.,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.15
    )

    train_generator = datagen.flow_from_directory(
        data_dir, target_size=(img_height, img_width), batch_size=batch_size,
        shuffle=True, subset='training', class_mode='categorical'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir, target_size=(img_height, img_width), batch_size=batch_size,
        shuffle=False, subset='validation', class_mode='categorical'
    )

    class_labels = list(train_generator.class_indices.keys())

    # Load pre-trained InceptionV3 model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(len(class_labels), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.001),
        ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=callbacks)

    y_test = validation_generator.classes
    y_pred = model.predict(validation_generator)
    y_pred = np.argmax(y_pred, axis=1)

    print(classification_report(y_test, y_pred, target_names=class_labels))

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an image classification model.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--model_save_path", type=str, required=True, help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()
    train_model(args.data_dir, args.model_save_path, args.epochs, args.batch_size, args.learning_rate)