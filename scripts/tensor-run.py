#!./tensorflow/bin/python

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model, load_model
import sys
import os 
import cv2
import numpy as np
from pathlib import Path
import fileinput

PROGRAM_NAME=str(sys.argv[0].lstrip('.').lstrip('/'))

def runModel(model, inputImage):
    file_path = Path(inputImage)

    if file_path.exists():
        # Load in the image from inputs from stdin 
        im = cv2.imread(inputImage, cv2.COLOR_BGR2RGB) 

        # We want to set the input image to the size of the images the network was 
        # trained on 
        dsize=(resize_width, resize_height)
        im = cv2.resize(im, dsize) 

        print(PROGRAM_NAME + ": info: It takes a long time to load the model...", file=sys.stderr)

        image_in_array = np.vstack([[im]])

        # Evaluate the model
        result = model.predict(image_in_array[:1])
        print(PROGRAM_NAME + "Result: " + str(result), file=sys.stderr)

        index=0
        indexOfHighestValue=-1
        highestValue=0
        # find index of the greatest value
        for probability in result[0]:
            if probability > highestValue:
                indexOfHighestValue = index
                highestValue = probability
            index = index + 1

        print(PROGRAM_NAME + ": info: indexOfHighestValue: " + str(indexOfHighestValue), file=sys.stderr)
        print(PROGRAM_NAME + ": info: size of classesArray: " + str(len(classesArray)), file=sys.stderr)
        if indexOfHighestValue >= 0:
            print("\"" + classesArray[indexOfHighestValue] + "\" identified")
        else:
            print("Could not identify tree")
    else:
        print(PROGRAM_NAME + ": warning: Could not find image: \'" + str(inputImage) + "\'", file=sys.stderr)

TrainingSetPath="Labels"
indexRecord="classes.txt.tmp"

resize_width=1000
resize_height=562
maxNum = 3
checkpoint_path = "TreeIdentifyTensorFlowModel.keras"

def get_model():
    # Create a simple model.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(resize_height, resize_width, 3)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(maxNum)
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
    return model


print(tf.__version__)

classesArray = []

inputImages = []

# Read in the classes file, which contains a record of the trees we trained on
file_path = indexRecord
try:
    with open(file_path, 'r') as file:
        for line in file:
            # Each 'line' variable will contain one line from the file,
            # including the newline character at the end (e.g., '\n').
            # You can process each line here.
            classesArray.append(line.strip())  # .strip() removes leading/trailing whitespace, including newlines
except FileNotFoundError:
    print(PROGRAM_NAME + ": error: The file '{file_path}' was not found.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(PROGRAM_NAME + ": error: An error occurred: {e}", file=sys.stderr)
    sys.exit(1)

model = tf.keras.models.load_model(checkpoint_path)
if model is not None:
    print(PROGRAM_NAME + ": info: Model loaded successfully!", file=sys.stderr)
else:
    print(PROGRAM_NAME + ": error: Failed to load the model.", file=sys.stderr)
    sys.exit(1)

# If no positional parameters are provided, take input from stdin
if len(sys.argv) < 2:
    for inputImage in fileinput.input():
        runModel(model, inputImage.rstrip())
else:
    inputImages = sys.argv[1:]
    for inputImage in inputImages:
        runModel(model, inputImage)

        
