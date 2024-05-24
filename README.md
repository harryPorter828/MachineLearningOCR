Uses the inftyCDB data set, which consists of the images in the 'Images' file and the InftyCDB-1.csv
file which describes the location of each character in each image in the 'Images' file.
'formatTrainingImages.py' and 'createTrainingData.py' produce an .npy file, from which 'createModel.py'
can be used to create a classification model which can classify 250 different characters.

Go to http://www.inftyproject.org/en/database.html to download the InftyCDB-1 database used for the project.
Transfer all images from the download to the 'images' foler and the csv file to the main project folder.

Key dependency : TensorFlow
I set up a tensorflow environement to run the code

