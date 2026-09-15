#!./tensorflow/bin/python

import tensorflow as tf
import sys
from os import walk
import cv2
import numpy as np
from libs.parse_annotation import *

PROGRAM_NAME=str(sys.argv[0].lstrip('.').lstrip('/'))

print(tf.__version__)

TrainingSetPathList=[
      "RillitoPark"
    , "CherryAvePark"
]

checkpoint_path = "TreeIdentifyTensorFlowModelCropped.keras"
indexRecordNew="classes-cropped.txt.tmp"
trainImageHeight=500
trainImageWidth=500

# classesArray is a list of trees
classesArray = []

## 1. Produce list of files 
# Parse the tagged files 
textFileList = []
jpgFileList = [] 
for TrainingSetPath in TrainingSetPathList:
    for (dirpath, dirnames, filenames) in walk(TrainingSetPath):
        for filename in filenames:
            lastPeriod=filename.rfind('.')
            if lastPeriod > 0:
                if filename[lastPeriod:].lower() == ".xml":
                    textFileList.append(TrainingSetPath + "/" + filename)
                elif filename[lastPeriod:].lower() == ".jpg":
                    jpgFileList.append(TrainingSetPath + "/" + filename)
                else:
                    print(PROGRAM_NAME + ": Warning: Unidentified extension: " + str(filename[lastPeriod:]) + " found on file path, " + str(filename) + ", continuing.", file=sys.stderr)
    
#print(str(textFileList))
#print(str(jpgFileList))
#sys.exit()    
   
# xyFileList contains a dictionary, associating image file with text file 
xyFileList = []
# Open each text file and get the first character before the first space and 
# put it in the y_train list 
y_train = [] 
x_train = []

## 2. Associate images with xml files 
# For each text file, associate xml with the associated image 
for x in textFileList:
    xLastPeriod=x.rfind('.')
    for y in jpgFileList:
        yLastPeriod=y.rfind('.')
        if x[:xLastPeriod] == y[:yLastPeriod]:
            xyFileList.append({'image': y, 'text':x, 'parsed': False})

# treeList holds a dictionary of the parameters
# 'xmin': x coordinate of upper left corner of bounding box
# 'ymin': y coordinate of upper left corner of bounding box
# 'xmax': x coordinate of lower right corner of bounding box
# 'ymax': y coordinate of lower right corner of bounding box
# 'treeID': index of tree type stored in classesArray
# 'imageID':  index of image stored in xyFileList

treeList = []

## 3. Parse XML
# Fetch the first value from the input file,
# x is a dictionary of 'image' and 'text'
# This loop parses the input files 
imageCounter = -1
for x in xyFileList: 
    imageCounter = imageCounter + 1
    xLastPeriod=x["text"].rfind('.')  
    parseList = []
    # When the tag file associated with the image is an xml file
    if x["text"][xLastPeriod:] == '.xml':
        success, width, height, depth, parseList = parseDronePicsXml(x["text"], classesArray)
        if not success:
            print(PROGRAM_NAME + ": Warning: Issue parsing: " + str(x["text"]) + " for " + str(x["image"]) + " skipping", file=sys.stderr)
            continue
    else:
        print(PROGRAM_NAME + ": Warning: No parser for extension: " + str(x["text"][xLastPeriod:]) + " for " + str(x["image"]) + " skipping", file=sys.stderr)
        continue
    #x['y_train_index'] = number_int
    
    for i in parseList:
        treeList.append({'xmin': i['xmin'], 'ymin': i['ymin'], 'xmax': i['xmax'], 'ymax': i['ymax'], 'treeID': i['classID'], 'imageID': imageCounter})
    x['parsed'] = True
                    

numTrees=len(treeList)
numTreeTypes=len(classesArray)

# x_train contains the list of images it is aligned with y_train, which is the
# list of arrays of indicies into class array. 
x_train = np.zeros(shape=(numTrees, trainImageHeight, trainImageWidth, 3), dtype=np.uint8)
y_train = np.zeros(shape=(numTrees, numTreeTypes))
## 4. Produce cropped images and put into x_train, which is the list of images
imageCounter = 0
for i in treeList:
    im = cv2.imread(xyFileList[i['imageID']]['image'], cv2.COLOR_BGR2RGB) 
    cropped_image = im[i['ymin']:i['ymax'], i['xmin']:i['xmax']] 
    resized_image = cv2.resize(cropped_image, (trainImageHeight, trainImageWidth)) 
    x_train[imageCounter] = resized_image
    a = [0.0] * len(classesArray)
    y_train[imageCounter][i['treeID']] = 1.0
    imageCounter = imageCounter + 1

# Build a machine learning model
# Sequential is useful for stacking layers where each layer has one input
# tensor and one output tensor. Layers are functions with a known mathmatical 
# structure that can be reused and have trainable variables. Most TensorFlow 
# models are composed of layers. This model uses the Flatten, Dense, and Dropout
# layers. 
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(
    32,
    (3,3),
    activation='relu',
    input_shape=(trainImageHeight, trainImageWidth,3)
  ),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(numTreeTypes, activation='softmax')
])

print(str(model.summary()))

# For each example, the model returns a vector of logits or log-odds scores, 
# one for each class. 
predictions = model(x_train[:1]).numpy()
predictions

#The tf.nn.softmax function converts these logits to probabilities for each class:
tf.nn.softmax(predictions).numpy()

# Define a loss function for training using losses.SparseCategoricalCrossentropy:
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

#The loss function takes a vector of ground truth values and a vector of logits and returns a scalar loss for each example. This loss is equal to the negative log probability of the true class: The loss is zero if the model is sure of the correct class.

# This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to -tf.math.log(1/10) ~= 2.3.
#print(str(y_train[0]))
#print(str(predictions[:1][0]))
loss_fn(y_train[:1], predictions).numpy()

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
    
except FileNotFoundError as e:
    print(PROGRAM_NAME + ": Error: writing to file: " + str(e), file=sys.stderr)
    sys.exit(1)
except PermissionError as e:
    print(PROGRAM_NAME + ": Error: writing to file: " + str(e), file=sys.stderr)
    sys.exit(1)
except OSError as e:
    print(PROGRAM_NAME + ": Error: writing to file: " + str(e), file=sys.stderr)
    sys.exit(1)
           
