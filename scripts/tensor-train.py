#!./tensorflow/bin/python

import tensorflow as tf
import sys
from os import walk
import cv2
import numpy as np
from libs.parse_annotation import *

PROGRAM_NAME=str(sys.argv[0].lstrip('.').lstrip('/'))

print(tf.__version__)

# Load the dataset, look in the 'Lables' directory and find the 
# DJI_0022.txt files and correlate them with the 
# DJI_0022.JPG files

TrainingSetPathList=[
    "Parker"
    , "RillitoPark"
    , "CherryAvePark"
]

checkpoint_path = "TreeIdentifyTensorFlowModel.keras"
indexRecord="Parker/classes.txt"
indexRecordNew="classes.txt.tmp"

# Amount to scale input images by per axis
imageScaleFactor=0.25
classesArray = []
classesFileFound = True

# Read in the classes file, which contains a record of the trees we trained on
file_path = indexRecord
try:
    with open(file_path, 'r') as file:
        classesFileFound = True
        for line in file:
            # Each 'line' variable will contain one line from the file,
            # including the newline character at the end (e.g., '\n').
            # You can process each line here.
            classesArray.append(line.strip())  # .strip() removes leading/trailing whitespace, including newlines
except FileNotFoundError:
    print(PROGRAM_NAME + ": Warning: classes file: " + indexRecord + " not found, if tagged with text format, these images will not be used to train.", file=sys.stderr)
    classesFileFound = False
except Exception as e:
    print(PROGRAM_NAME + ": Warning: classes file: " + indexRecord + " not found, if tagged with text format, these images will not be used to train., exception: " + str(e), file=sys.stderr)
    classesFileFound = False
    
initialClassArrayLen = len(classesArray)

textFileList = []
jpgFileList = [] 
for TrainingSetPath in TrainingSetPathList:
    for (dirpath, dirnames, filenames) in walk(TrainingSetPath):
        for filename in filenames:
            lastPeriod=filename.rfind('.')
            if lastPeriod > 0:
                if filename[lastPeriod:].lower() == ".txt":
                    textFileList.append(TrainingSetPath + "/" + filename)
                if filename[lastPeriod:].lower() == ".xml":
                    textFileList.append(TrainingSetPath + "/" + filename)
                elif filename[lastPeriod:].lower() == ".jpg":
                    jpgFileList.append(TrainingSetPath + "/" + filename)

# xyFileList contains a dictionary, associating image file with text file 
xyFileList = []

for x in textFileList:
    xLastPeriod=x.rfind('.')
    for y in jpgFileList:
        yLastPeriod=y.rfind('.')
        if x[:xLastPeriod] == y[:yLastPeriod]:
            xyFileList.append({'image': y, 'text':x, 'parsed': False})


# Open each text file and get the first character before the first space and 
# put it in the y_train list 
y_train = [] 
maxNum = 0

# Fetch the first value from the input file 
for x in xyFileList: 
    number_str = ""
    xLastPeriod=x["text"].rfind('.')
    # When the tag file associated with the image is a txt file
    if x["text"][xLastPeriod:] == '.txt':
        if not classesFileFound:
            print(PROGRAM_NAME + ": Error: Not training with: " + str(x["image"]) + " because classes file not found.", file=sys.stderr)
            break 
        try:
            with open(x["text"], 'r') as f:
                while True:
                    c = f.read(1)
                    if not c or c == ' ':
                        break
                    else:
                        number_str = number_str + c
        except FileNotFoundError:
            print(PROGRAM_NAME + ": Warning: classes file: " + str(x["text"]) + " not found, skipping.", file=sys.stderr)
            continue
        except Exception as e:
            print(PROGRAM_NAME + ": Warning: classes file: " + str(x["text"]) + " not found, skipping.", file=sys.stderr)
            continue
    # When the tag file associated with the image is an xml file
    elif x["text"][xLastPeriod:] == '.xml':
        success, width, height, depth, treeList = parseDronePicsXml(x["text"], classesArray)
        if not success:
            print(PROGRAM_NAME + ": Warning: Issue parsing: " + str(x["text"]) + " for " + str(x["image"]) + " skipping", file=sys.stderr)
            continue
        number_str = treeList[0]['classID']
    else:
        print(PROGRAM_NAME + ": Warning: No parser for extension: " + str(x["text"][xLastPeriod:]) + " for " + str(x["image"]) + " skipping", file=sys.stderr)
        continue
    number_int = int(number_str)
    #x['y_train_index'] = number_int
    x['y_train'] = number_int
    x['parsed'] = True
    y_train.append(number_int)
    if number_int > maxNum:
        maxNum = number_int

# Set the index size for later on 
maxNum = maxNum + 1
imgHeight = 0
imgWidth = 0

# Use OpenCV NumPy interface, gotten from: 
# https://stackoverflow.com/questions/7762948/how-to-convert-an-rgb-image-to-numpy-array
for x in xyFileList:
    imageFilePath = x["image"]
    print(PROGRAM_NAME + ": Info: Reading in: " + imageFilePath, file=sys.stderr)
    im = cv2.imread(imageFilePath, cv2.COLOR_BGR2RGB) 
    # scale image to be 1/16 the size
    im = cv2.resize(im, (0,0), fx=imageScaleFactor, fy=imageScaleFactor) 
    
    # Check to make sure all images are the same size 
    if imgHeight == 0:
        imgHeight = np.size(im, 0)
        imgWidth = np.size(im, 1)
    elif np.size(im, 0) != imgHeight or imgWidth != np.size(im, 1):
        print(PROGRAM_NAME + ": Error:  is not like the rest " + str(np.size(im, 0) + " != " + str(imgHeight) + " or " + str(np.size(im, 1)) + " != " + str(imgWidth)), file=sys.stderr)
        sys.exit(1)

    x["x_train"] = im
    #print(str(x["x_train"]))

    
print("Height: " + str(imgHeight) + " Width: " + str(imgWidth))

# Create the y_train array, it is an array of arrays of floats, the size of 
# the second array is the number of different trees we're identifying. 
# Each index represents a probability of it being the designated tree in the 
# classes.txt file 
#for x in xyFileList: 
#    array = np.zeros(maxNum, dtype=float)
#    array[x['y_train_index']] = 1.0
#    x["y_train"] = array

# Populate the x train and y train arrays 
count = 0
for n in xyFileList:
    if n['parsed']:
        break
    count = count + 1

x_train = np.vstack([[xyFileList[count]["x_train"]]])
y_train = np.array(y_train)

for n in xyFileList[count+1:]:
    if n['parsed']:
        x_train = np.vstack([x_train, [n["x_train"]]])

# Verify dimensions 
if y_train.ndim != 1:
    print(PROGRAM_NAME + ": Error: y_train.ndim not 1, it is: " + str(y_train.ndim), file=sys.stderr)
    sys.exit(1)
if x_train.ndim != 4:
    print(PROGRAM_NAME + ": Error: x_train.ndim not 1, it is: " + str(x_train.ndim), file=sys.stderr)
    sys.exit(1)

if len(y_train) != len(x_train):
    print(PROGRAM_NAME + ": Error: Length of training record : " + str(len(y_train)) + " does not equal length of image set: " + str(len(x_train)), file=sys.stderr)
    sys.exit(1)

# Build a machine learning model
# Sequential is useful for stacking layers where each layer has one input
# tensor and one output tensor. Layers are functions with a known mathmatical 
# structure that can be reused and have trainable variables. Most TensorFlow 
# models are composed of layers. This model uses the Flatten, Dense, and Dropout
# layers. 
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(imgHeight, imgWidth, 3)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(maxNum)
])

print(str(model.summary()))

# For each example, the model returns a vector of logits or log-odds scores, 
# one for each class. 
predictions = model(x_train[:1]).numpy()
predictions

#The tf.nn.softmax function converts these logits to probabilities for each class:
tf.nn.softmax(predictions).numpy()

# Define a loss function for training using losses.SparseCategoricalCrossentropy:
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#The loss function takes a vector of ground truth values and a vector of logits and returns a scalar loss for each example. This loss is equal to the negative log probability of the true class: The loss is zero if the model is sure of the correct class.

# This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to -tf.math.log(1/10) ~= 2.3.
loss_fn(y_train[:1], predictions[0]).numpy()

# Before you start training, configure and compile the model using Keras Model.compile. Set the optimizer class to adam, set the loss to the loss_fn function you defined earlier, and specify a metric to be evaluated for the model by setting the metrics parameter to accuracy.
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

     
# Use the Model.fit method to adjust your model parameters and minimize the loss:
model.fit(x_train, y_train, epochs=5)

# The Model.evaluate method checks the model's performance, usually on a validation set or test set.
# model.evaluate(x_test,  y_test, verbose=2)

# The image classifier is now trained to ~98% accuracy on this dataset. To learn more, read the TensorFlow tutorials.

# If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it:
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

#Congratulations! You have trained a machine learning model using a prebuilt dataset using the Keras API.

# For more examples of using Keras, check out the tutorials. To learn more about building models with Keras, read the guides. If you want learn more about loading and preparing data, see the tutorials on image data loading or CSV data loading.
#probability_model(x_test[:5])

model.save(checkpoint_path)

# Take the classesArray that we have built up and write it out to a temporary file
try:             
    with open(indexRecordNew, "w") as f:
        try:
            # save the classes array
            for entry in classesArray:
                f.write(str(entry) + "\n")

        except IOError as e:
            print(PROGRAM_NAME + ": Error: writing to file: " + str(e), file=sys.stderr)
            sys.exit(1)
        except OSError as e:
            print(PROGRAM_NAME + ": Error: writing to file: " + str(e), file=sys.stderr)
            sys.exit(1)
    
    if initialClassArrayLen < len(classesArray):
        print(PROGRAM_NAME + ": Info: classes file: " + str(indexRecordNew) + " has new records added to it. As a developer, please integrate and update these changes to: " + str(indexRecord) + " so that when running the model, users can know what trees have been identified by the model based on ID.", file=sys.stderr)
    
except FileNotFoundError as e:
    print(PROGRAM_NAME + ": Error: writing to file: " + str(e), file=sys.stderr)
    sys.exit(1)
except PermissionError as e:
    print(PROGRAM_NAME + ": Error: writing to file: " + str(e), file=sys.stderr)
    sys.exit(1)
except OSError as e:
    print(PROGRAM_NAME + ": Error: writing to file: " + str(e), file=sys.stderr)
    sys.exit(1)


    


