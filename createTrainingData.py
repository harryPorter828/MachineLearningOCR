import os
import pandas as pd
from PIL import Image, ImageFile
from matplotlib.pyplot import imshow
import requests
from io import BytesIO
import numpy as np


def main():
    """
    Takes all training images created in 'formatTrainingImages.py'
    and converts them into a npArray to be used for training
    """
    imageList = []
    training_data = []

    df = pd.read_csv("InftyCDB-1.csv", encoding="latin1")  # open table with data
    sample = df
    size = len(df)

    for i in range(size):
        imageList.append("TrainingImages\\" + str(i) + ".png")
    j = 0
    for pic in imageList:
        img = Image.open(pic)
        training_data.append(np.asarray(img))
        j += 1
        print(j)
    training_data
    np.save("training", training_data)  # Saves as "training.npy"
    print("Done.")

if __name__ == "__main__":
    main()
