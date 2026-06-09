#!./tensorflow/bin/python

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model, load_model
import sys
import os 
import cv2
import numpy as np
from pathlib import Path
import argparse

PROGRAM_NAME=str(sys.argv[0].lstrip('.').lstrip('/'))

trainImageHeight=500
trainImageWidth=500
current_dir = os.getcwd()
indexRecord=current_dir + "/classes-cropped.txt.tmp"
classesArray = []

def runModel(model, inputLine, verbose=0):
    splitline = inputLine.split()
    
    inputImage = splitline[0]
    x=int(0)
    width=int(0)
    y=int(0)
    height=int(0)
    
    if len(splitline) >= 5:
        x=int(splitline[1])
        width=int(splitline[2])
        y=int(splitline[3])
        height=int(splitline[4])
    
    file_path = Path(inputImage)

    if file_path.exists():
        # Load in the image from inputs from stdin 
        im = cv2.imread(inputImage, cv2.COLOR_BGR2RGB) 

        if width > 0 and height > 0:
            im = im[y:(y+height), x:(x+width)] 
        im = cv2.resize(im, (trainImageHeight, trainImageWidth)) 

        image_in_array = np.vstack([[im]])

        # Evaluate the model
        result = model.predict(image_in_array[:1], verbose=0)
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
        
        if verbose > 0: 
            print(PROGRAM_NAME + ": info: indexOfHighestValue: " + str(indexOfHighestValue), file=sys.stderr)
            print(PROGRAM_NAME + ": info: size of classesArray: " + str(len(classesArray)), file=sys.stderr)
        if indexOfHighestValue >= 0:
            print("\"" + classesArray[indexOfHighestValue] + "\" identified ", end='')
            #print(str(result[0]).replace('\n',''), end='')
            if len(splitline) > 5:
                print(" " + str(splitline[5]).replace('\n',''))
            else:
                print()
        else:
            print("Could not identify tree, ", end='')
    else:
        print(PROGRAM_NAME + ": warning: Could not find image: \'" + str(inputImage) + "\'", file=sys.stderr)
        
    return 0

parser = argparse.ArgumentParser()
parser.add_argument('filenames', nargs='*') 
parser.add_argument('--verbose', '-v', action='count', default=0)
args = parser.parse_args()

TrainingSetPath="Labels"
checkpoint_path = current_dir + "/TreeIdentifyTensorFlowModelCropped.keras"

print(PROGRAM_NAME + ": info: version: " + tf.__version__, file=sys.stderr)

print(indexRecord, file=sys.stderr)
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

print(PROGRAM_NAME + ": info: It takes a long time to load the model...", file=sys.stderr)
model = tf.keras.models.load_model(checkpoint_path)
if model is not None:
    print(PROGRAM_NAME + ": info: Model loaded successfully!", file=sys.stderr)
else:
    print(PROGRAM_NAME + ": error: Failed to load the model.", file=sys.stderr)
    sys.exit(1)

# If no positional parameters are provided, take input from stdin
if len(args.filenames) == 0:
    while 1:
        line = sys.stdin.readline()
        if not line: break
        runModel(model, line, args.verbose)
        
# If positional arguments are provided 
else:
    for inputImage in args.filenames:
        runModel(model, inputImage, args.verbose)

        
