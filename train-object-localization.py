#!./tensorflow/bin/python

import sys
from os import walk
import numpy as np
import tensorflow as tf
import cv2
import keras_cv
from libs.parse_annotation import *

PROGRAM_NAME=str(sys.argv[0].lstrip('.').lstrip('/'))

# 1. Define your hyperparameters
NUM_CLASSES = 1 # Just "tree"
BATCH_SIZE = 4
EPOCHS = 20
 
trainImageHeight=512
trainImageWidth=512
maxNumTreesPerImage=10
 
# 2. Prepare your dataset 
# (You will need to parse your local XML/JSON into this format)
def load_dataset(TrainingSetPathList=["RillitoPark", "CherryAvePark"]):
    # Placeholder: Replace with your actual image loading & label parsing logic
    # Images shape: (B, H, W, 3), BBoxes shape: (B, M, 4) where M is max boxes per image
    
    ## 2.1. Produce list of files 
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

    ## 2.2. Associate images with xml files 
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
    # classesArray is a list of trees
    classesArray = []

    ## 2.3. Parse XML
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
    numImages=len(xyFileList)
    
    images = np.zeros(shape=(numImages, trainImageHeight, trainImageWidth, 3), dtype=np.uint8)
    classes = np.zeros(shape=(numImages, maxNumTreesPerImage))
    boxes = np.zeros(shape=(numImages, maxNumTreesPerImage, 4))
    
    # build up boxes
    imageCounter = 0
    imageIDArray = [0]*numImages
    for tree in treeList:
        imageCounter = tree['imageID']
        if imageIDArray[imageCounter] < maxNumTreesPerImage: 
            classes[imageCounter][imageIDArray[imageCounter]] = 1
            boxes[imageCounter][imageIDArray[imageCounter]][0] = tree['xmin']
            boxes[imageCounter][imageIDArray[imageCounter]][1] = tree['ymin']
            boxes[imageCounter][imageIDArray[imageCounter]][2] = tree['xmax']
            boxes[imageCounter][imageIDArray[imageCounter]][3] = tree['ymax']
            imageIDArray[imageCounter] = imageIDArray[imageCounter] + 1
        else:
            continue
        
    ## 4. Produce cropped images and put into x_train, which is the list of images
    imageCounter = 0
    for x in xyFileList:
        im = cv2.imread(x['image'], cv2.COLOR_BGR2RGB) 
        resized_image = cv2.resize(im, (trainImageHeight, trainImageWidth)) 
        images[imageCounter] = resized_image
        imageCounter = imageCounter + 1
    
    #images = np.random.uniform(0, 255, (20, 512, 512, 3)).astype("float32")
    #boxes = np.random.uniform(10, 400, (20, maxNumTreesPerImage, 4)).astype("float32")
    #classes = np.zeros((numTrees, maxNumTreesPerImage), dtype="float32") # Class 0 for all boxes
    
    bbox_dict = {"boxes": boxes, "classes": classes}
    dataset = tf.data.Dataset.from_tensor_slices((images, bbox_dict))
    return dataset.batch(BATCH_SIZE)
 
train_ds = load_dataset()
 
# 3. Load a pre-trained Backbone and Object Detection Model
# We use ResNet50 backbone initialized with COCO weights for a massive head start
backbone = keras_cv.models.ResNet50Backbone.from_preset("resnet50_imagenet")
model = keras_cv.models.RetinaNet(
    num_classes=NUM_CLASSES,
    bounding_box_format="xyxy", # Change to match your coordinate system
    backbone=backbone,
)
 
# 4. Compile the model with focal loss (standard for object detection)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(
    classification_loss="focal",
    box_loss="smoothL1",
    optimizer=optimizer
)
 
# 5. Train!
model.fit(train_ds, epochs=EPOCHS)
 
# 6. Save your new custom model
model.save("custom_tree_detector.keras")
