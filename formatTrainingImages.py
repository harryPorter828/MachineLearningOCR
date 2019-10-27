"""
Goes through the Infty CDB and scrapes characters based on their location as described by
the InftyCDB-1.csv file. Saves each individual character to a png file of size 64x64 pixels
"""


import os
import pandas as pd
from PIL import Image, ImageFile
from matplotlib.pyplot import imshow
import requests
from io import BytesIO
import numpy as np


def make_square(img):
    """
    crops the npArray as described in 'saveImage' to a square shape
    """
    cols, rows = img.size

    if rows > cols:
        pad = (rows - cols) / 2
        img = img.crop((0, 0, cols, cols))
    else:
        pad = (cols - rows) / 2
        img = img.crop((0, 0, rows, rows))
    return img


def saveImage(down, up, right, left, matrixRepresentation, name):
    """
    Takes the coordinates of where the character is stored as well as the matrix 'npArray' of
    the sheet where it's located. Picks out the individual character from it's coordinates by
    indexing the npArray
    Shapes the image to a standard 64x64 pixel image and saves it
    """
    height = down - up
    width = right - left
    imSize = max(height, width)
    newChar = np.ones((imSize, imSize), dtype=np.uint8)
    for i in range(up, down):
        for j in range(left, right):
            newChar[i - up + int(round(((abs(height - imSize)) / 2) - 1))][
                j - left + int(round(((abs(width - imSize)) / 2) - 1))
            ] = matrixRepresentation[i][j]
    newCharIm = Image.fromarray(newChar * 255)
    newCharIm = newCharIm.resize((64, 64), Image.ANTIALIAS)
    newCharIm.save(("TrainingImages\\" + name + ".png"), "PNG")
    print(name + ".png saved!")


def main():
    df = pd.read_csv("InftyCDB-1.csv", encoding="latin1")

    sample = df
    size = len(df)
    """
    .png files containing characters to be used for training
    can be indexed based off of a sheet and journal Id
    """
    sheets = sample["SheetId"].values
    journals = sample["JournalId"].values
    pages = []

    """
    Takes the ordered Sheet and Journal Id's to build the names of the .png
    files containing characters to be used for training
    """

    for i in range(size):
        pages.append(str(sheets[i]))
    for i in range(size):
        if len(pages[i]) < 3:
            while len(pages[i]) < 3:
                pages[i] = "0" + (pages[i])
        pages[i] = str(journals[i]) + "_" + (pages[i])
        if len(pages[i]) < 6:
            while len(pages[i]) < 6:
                pages[i] = "0" + (pages[i])
    leftCoord = sample["left"].values
    rightCoord = sample["right"].values
    upCoord = sample["top"].values
    downCoord = sample["bottom"].values  # gets coordinates of characters
    """
    Creates a 2D array of the coordinates of each individual
    character's position for each sheet of every journal
    """
    pixelIteration = []
    for i in range(size):
        pixelIteration.append([])
    for i in range(size):
        pixelIteration[i].append(downCoord[i])
        pixelIteration[i].append(upCoord[i])
        pixelIteration[i].append(rightCoord[i])
        pixelIteration[i].append(leftCoord[i])
    """
    For each character, open the journal and sheet where it is located
    Save character by locating it based on its coordinates stored in
    'pixelIteration'
    """
    for i in range(size):

        image = Image.open("Images\\" + (pages[i]) + ".png")
        matrixRepresentation = np.asarray(image)
        saveImage(
            pixelIteration[i][0],
            pixelIteration[i][1],
            pixelIteration[i][2],
            pixelIteration[i][3],
            matrixRepresentation,
            str(i),
        )


if __name__ == "__main__":
    main()
