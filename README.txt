Uses the inftyCDB data set, which consists of the images in the 'Images' file and the InftyCDB-1.csv
file which describes the location of each character in each image in the 'Images' file.
'formatTrainingImages.py' and 'createTrainingData.py' produce an .npy file, from which 'createModel.py'
can be used to create a classification model which can classify 250 different characters.

Run 'InftyCDB-1.EXE' to download images used to get training data. Move csv file
created from running 'InftyCDB-1.EXE'

Key dependency : TensorFlow
I set up a tensorflow environement to run the code

