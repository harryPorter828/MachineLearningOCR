"""
Creates a model which can be used to classify characters which
appear in mathematical documents
"""

import os
import pandas as pd
from PIL import Image, ImageFile
from matplotlib.pyplot import imshow
import requests
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random


def main():
    df = pd.read_csv("InftyCDB-1.csv", encoding="latin1")  # open table with data
    size = len(df)

    sliceStart = random.randint(0, 550000)

    x_train = np.load("training.npy")
    x_train1, x_test, x_train2 = np.split(
        x_train, [sliceStart, sliceStart + 50000]
    )  # creates a random split for training and test data
    x_train = np.concatenate((x_train1, x_train2), axis=0)

    df = pd.read_csv("InftyCDB-1.csv", encoding="latin1")
    df1 = df[0:sliceStart]
    df2 = df[(sliceStart + 50000) : size]
    y_train = pd.concat([df1, df2])["Entity"].values

    df = pd.read_csv("InftyCDB-1.csv", encoding="latin1")
    y_test = df[sliceStart : sliceStart + 50000]["Entity"].values

    """
    Converts all pixel values in the training data to between 0 and 1
    so they can be more easily processed by the model
    """
    for i in range(len(x_train)):
        x_train[i] = x_train[i] / 255.0
    for i in range(len(x_test)):
        x_test[i] = x_test[i] / 255.0
    newChar = x_test[0]
    newCharIm = Image.fromarray(newChar * 255)
    newCharIm.show()
    """
    Creates labels to classify data,
    'unique' used to check that label isn't stored twice
    """
    unique = {}
    for i in df["Entity"].values:
        if i not in unique:
            unique[i] = len(unique)
    print(unique)
    """
    splits labels into test and training labels
    """
    for i in range(len(y_train)):
        y_train[i] = unique[y_train[i]]
    for i in range(len(y_test)):
        y_test[i] = unique[y_test[i]]
    """
    Creates the layers for the model to be trained
    """
    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(64, 64)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(250, activation=tf.nn.softmax),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(x_train, y_train, epochs=5)

    test_loss, test_acc = model.evaluate(x_test, y_test)

    """
    Checks the accuracy of the model based on the test data
    """
    print("Test accuracy:", test_acc)
    print("Test lost:", test_loss)
    model.save("OCRMaths.h5")  # saves both the nodes and the weights in the model


if __name__ == "__main__":
    main()
