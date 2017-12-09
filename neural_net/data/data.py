import numpy as np
import fer_csv as FER

class Data:
    max_num_pixels = 2304
    data_sources   = [
        FER.FER_CSV()
    ]

    def __init__(self, print_progress = False):
        self.print_progress = print_progress
        self.Xtrain         = None
        self.Ytrain         = None
        self.Xdev           = None
        self.Ydev           = None
        self.Xtest          = None
        self.Ytest          = None

    def load_data(self):
        for data_source in self.data_sources:
            data_source.load_data(self.print_progress)

            if self.Xtrain == None:
                self.Xtrain = data_source.Xtrain
                self.Ytrain = data_source.Ytrain

                self.Xdev = data_source.Xdev
                self.Ydev = data_source.Ydev

                self.Xtest = data_source.Xtest
                self.Ytest = data_source.Ytest
            else:
                self.Xtrain = np.column_stack((self.Xtrain, data_source.Xtrain))
                self.Ytrain = np.column_stack((self.Ytrain, data_source.Ytrain))

                self.Xdev = np.column_stack((self.Xdev, data_source.Xdev))
                self.Ydev = np.column_stack((self.Ydev, data_source.Ydev))

                self.Xtest = np.column_stack((self.Xtest, data_source.Xtest))
                self.Ytest = np.column_stack((self.Ytest, data_source.Ytest))


d = Data(True)
d.load_data()
print('Xtrain is a', type(d.Xtrain))
print('Xtrain.shape is', d.Xtrain.shape)
print('Xtrain is', d.Xtrain)
print('Ytrain is a', type(d.Ytrain))
print('Ytrain.shape is', d.Ytrain.shape)
print('Ytrain is', d.Ytrain)

print('Xdev is a', type(d.Xdev))
print('Xdev.shape is', d.Xdev.shape)
print('Xdev is', d.Xdev)
print('Ydev is a', type(d.Ydev))
print('Ydev.shape is', d.Ydev.shape)
print('Ydev is', d.Ydev)

print('Xtest is a', type(d.Xtest))
print('Xtest.shape is', d.Xtest.shape)
print('Xtest is', d.Xtest)
print('Ytest is a', type(d.Ytest))
print('Ytest.shape is', d.Ytest.shape)
print('Ytest is', d.Ytest)
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
# into the script and unroll it into the X matrix. Eg:
# from PIL import Image
# import os
# for filename in os.listdir('../data/facial-recognition/google_image_search/happy'):
#     print(filename)
# test = Image.open("../data/facial-recognition/IMFDB_final/ANR/Missamma/images/ANR_1.jpg")

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
