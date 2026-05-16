#!./tensorflow/bin/python

# harvest-images.py takes a path, finds all xml files, for each drone pic in 
# the xml file, return each image path, its bounding box, and type of tree it 
# is annotated as. 

import sys
from os import walk
from libs.parse_annotation import *

textFileList = []
jpgFileList = [] 

PROGRAM_NAME=str(sys.argv[0].lstrip('.').lstrip('/'))

if len(sys.argv) < 2:
    print(PROGRAM_NAME + ": ERROR: No input parameter provided. Please provide a path to a directory.", file=sys.stderr)
    sys.exit(1)

for validationPath in sys.argv[1:]:
    for (dirpath, dirnames, filenames) in walk(validationPath):
        for filename in filenames:
            lastPeriod=filename.rfind('.')
            if lastPeriod > 0:
                if filename[lastPeriod:].lower() == ".xml":
                    textFileList.append(validationPath + "/" + filename)
                elif filename[lastPeriod:].lower() == ".jpg":
                    jpgFileList.append(validationPath + "/" + filename)
                else:
                    print(PROGRAM_NAME + ": Warning: Unidentified extension: " + str(filename[lastPeriod:]) + " found on file path, " + str(filename) + ", continuing.", file=sys.stderr)
        
# xyFileList contains a dictionary, associating image file with text file 
xyFileList = []
# Open each text file and get the first character before the first space and 
# put it in the y_train list 
y_train = [] 
x_train = []

# For each text file, associate xml with the associated image 
for x in textFileList:
    xLastPeriod=x.rfind('.')
    for y in jpgFileList:
        yLastPeriod=y.rfind('.')
        if x[:xLastPeriod] == y[:yLastPeriod]:
            xyFileList.append({'image': y, 'text':x, 'parsed': False})

speciesList=[]

for image in xyFileList:
    success, width, height, depth, returnList = parseDronePicsXml(image['text'], speciesList)
    if success:
        for tree in returnList:
            print(image['image'] + " " + str(tree['xmin']) + " " + str(tree['ymin']) + " " + str(tree['xmax'] - tree['xmin']) + " " + str(tree['ymax'] - tree['ymin']) + " " + tree['name'])

