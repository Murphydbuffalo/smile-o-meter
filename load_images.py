from PIL import Image
import os
import numpy as np
import csv

class FER_CSV:
    filename = '../data/facial-recognition/fer2013/fer2013.csv'

    def __init__(self):
        self.csv = csv.DictReader(open(self.filename))
        self.X   = []
        self.Y   = []

    def __label(self, row):
        # For the Smile-O-Meter we want three classes: 0 = Netural, 1 = Happy,   2 = Sad
        # Need to convert from the FER labels of:      0 = Angry,   1 = Disgust, 2 = Fear, 3 = Happy, 4 = Sad, 5 = Surprise, 6 = Neutral
        fer_label = int(row['emotion'])

        if fer_label == 3:
            label = 1
        elif fer_label < 5:
            label = 2
        else:
            label = 0

        return label

    def __pixels(self, row):
        return np.array(row['pixels'].split(), 'int')

    def load_data(self, printProgress = False):
        for index, row in enumerate(self.csv):
            label  = self.__label(row)
            pixels = self.__pixels(row)

            if printProgress == True and index % 100 == 0:
                print("label is", label)
                print("shape of pixels is", pixels.shape)
                print("pixels are", pixels)

            self.X.append(pixels)
            self.Y.append(label)

        self.X = np.array(self.X)
        self.Y = np.reshape(np.array(self.Y), (-1, 1))

        return True

fer_csv = FER_CSV()
fer_csv.load_data(True)
print('X is a', type(fer_csv.X))
print('X.shape is', fer_csv.X.shape)
print('Y is a', type(fer_csv.Y))
print('Y.shape is', fer_csv.Y.shape)


# for filename in os.listdir('../data/facial-recognition/google_image_search/happy'):
#     print(filename)
# test = Image.open("../data/facial-recognition/IMFDB_final/ANR/Missamma/images/ANR_1.jpg")
#
# arr  = np.array(test)
# print("array is", arr)
# print("reshaped array is", np.reshape(arr, -1))

# Data Wranglin'
# We've got data from various sources here, totaling 47,156 without IMFDB, and
# 81,688 with IMFDB. Going to scale every image down to 48x48 and grey them out.
#
# IMFDB - 34,512 images. The images from this dataset are very small and vary in size.
# Going to leave this data out for the initial training to keep the image sizes
# at 48x48 (the size of the FER images). Will add this data in if results aren't
# good enough without it.
#
# To label this data we need to recursively iterate through every directory (they
# are nested), locate the .txt file which contains a list of images in the directory
# and emotional labels (NEUTRAL, SADNESS, HAPPINESS, FEAR), and load each image into
# Python, add a 1 to the Y vector if the label is HAPPINESS, and then load the image
# into the script and unroll it into the X matrix.
#
# FER (Facial Expression Recognition Challenge) - 28,709 training examples, 3,589
# dev and test examples (35,887 total). You may want to adjust this train/dev/test split. You'll
# need to devise the appropriate split for your other sources of data as well.
# The data is all contained within a CSV with two columns: Emotion (integer from
# 1 to 6: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral), pixels
# (each example is 48x48 greyscale, you should grey out your other data set's images
# too). You want to iterate through the CSV, set the Y vector value to 1 if the
# expression label is 3 (happy), and then convert the second column string into
# a column in the X matrix.
#
# Google Image Search -  520 images (can get a few hundred more pretty easily if need be)
# Images are about 300x150 on average (all seem larger than 100x100).
# You hand labeled these, putting smile and non-smile images in separate directories.
# So just load the images in, set the Y vector value based on the directory, and unroll
# the image into the X matrix.
#
# JAFFE - 214 images @ 256x256 pixels. Labels for each image in a CSV (located
# within the README), which rates happiness (and other emotions) on a scale from
# 1 (low probability) to 5 (high probability). Will need to iterate through each
# row in the CSV, add an element to the Y vector (1 if rating is 3 or higher, else 0),
# load in the corresponding image, and unroll it into the X matrix.
#
# AMFED - 10,535 labeled video frames @ 320x240 pixels. Labels for each frame are
# in CSVs for every video file. Smile probability given from 0.0 to 1.0
# For every file will need to read each row of the corresponding CSV, get the
# time of the frame (first column), smile likelihood (second column), consider the
# frame to have a smile if that probability is 0.67 or greater, and use
#`ffmpeg -i inputfile -ss 00:00:55.000 -r 14 -vframes 1 outputfile{smile/no-smile}.jpg`
# to output an image of the frame with a filename that labels it as smile or no-smile.
# Then we can load these images into Python, and construct the X and Y matrices
# based on the pixels values and label (from the filename).
#
# Q: Better to exclude the large number of IMFDB images so we can keep the resolution
# of the remaining images at 100x100?
# Or better to have more examples with fewer features?
# ~45,000 images without IMFDB
# IMFDB has an additional ~35,000 images
# Try both! It'll be interesting to see if N or M matters more in this case.
#
# The lfw dataset isn't labeled by emotion. You can have AWS Rekognition label
# the images for you. Would cost about $35.
#
# Keep webcam images separate for dev and test sets
# Take remaining images and divide between train, dev, and test
# Then add webcam evenly to dev and test
#   => maybe 85/7.5/7.5 split? 30,250, 3,375, 3,375
# Resize images so they're all the same resolution (same number of features)
# Convert to numpmy arrays, reshape to column vectors
# Normalize to mean 0 and variance 1
# Shuffle
# This'll give you an NxM array of data
#
# How do you get the corresponding Mx1 row vector of labels (Y)?
# Add the label to the filename and update the correspondig entry in Y based on that?
# Regardless, you're going to need shell scripts for labeling data
# Probably will want to get all of your images into training/dev/test directories
# For easy loading into your Python scripts for loading & normalizing the data.
#
# Got access to the AMFED database.
# This is a collection of 243 videos, with 10,535 labeled frames
# So you might have around 15,000 examples total
