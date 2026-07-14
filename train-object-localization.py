#!./tensorflow/bin/python

import sys
from os import walk
import numpy as np
import keras
import tensorflow as tf
import cv2
import keras_cv
from libs.parse_annotation import *

PROGRAM_NAME=str(sys.argv[0].lstrip('.').lstrip('/'))

# 1. Define your hyperparameters
NUM_CLASSES = 2 # Just "tree" (1) and "not tree" (0)
BATCH_SIZE = 4
EPOCHS = 20
 
trainImageHeight=512
trainImageWidth=512
maxNumTreesPerImage=10

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512


# load_image will load an image into tf given the path to the image. 
def load_image(filename, imageHeight=IMAGE_HEIGHT, imageWidth=IMAGE_WIDTH):
    image = tf.io.read_file(filename)

    # Decode JPEG into RGB
    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    # Convert uint8 -> float32 in [0,1]
    image = tf.image.convert_image_dtype(
        image,
        tf.float32
    )

    # Resize
    image = tf.image.resize(
        image,
        (imageHeight, imageWidth)
    )

    return image
 
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
    
    images = np.zeros(shape=(numImages, trainImageHeight, trainImageWidth, 3))
    classes = np.zeros(shape=(numImages, maxNumTreesPerImage))
    boxes = np.zeros(shape=(numImages, maxNumTreesPerImage, 4))
    
    # build up boxes
    # There are one of more boxes per image 
    imageCounter = 0
    fileIdArray = [0]*numImages
    for tree in treeList:
        imageCounter = tree['imageID']
        if fileIdArray[imageCounter] < maxNumTreesPerImage: 
            classes[imageCounter][fileIdArray[imageCounter]] = 1
            boxes[imageCounter][fileIdArray[imageCounter]][0] = tree['xmin']
            boxes[imageCounter][fileIdArray[imageCounter]][1] = tree['ymin']
            boxes[imageCounter][fileIdArray[imageCounter]][2] = tree['xmax']
            boxes[imageCounter][fileIdArray[imageCounter]][3] = tree['ymax']
            fileIdArray[imageCounter] = fileIdArray[imageCounter] + 1
        else:
            continue
        
    ## 4. Produce cropped images and put into x_train, which is the list of images
    imageCounter = 0
    for x in xyFileList:
        loadedImage = load_image(x['image'])
        images[imageCounter] = loadedImage
        imageCounter = imageCounter + 1
    
    #images = np.random.uniform(0, 255, (20, 512, 512, 3)).astype("float32")
    #boxes = np.random.uniform(10, 400, (20, maxNumTreesPerImage, 4)).astype("float32")
    #classes = np.zeros((numTrees, maxNumTreesPerImage), dtype="float32") # Class 0 for all boxes
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            images,
            {
                "boxes": boxes,
                "classes": classes,
            }
        )
    )

    dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset
 
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
 
sample = next(iter(train_ds))
#print(sample)
#print(tf.nest.map_structure(lambda x: x.shape, sample))
#print(sample.keys())
#print(sample["images"].shape)
#print(sample["bounding_boxes"]["boxes"].shape)
#print(sample["bounding_boxes"]["classes"].shape)

#print("TensorFlow:", tf.__version__)
#print("Keras:", keras.__version__)
#print("KerasCV:", keras_cv.__version__)

# Debug outputs
#print(type(model))
#print(model.inputs)
#print(model.input_shape)

#print(type(backbone))
#print(backbone.inputs)
#print(backbone.input_shape)

# Debug outputs 
#sample = next(iter(train_ds))

#images, labels = sample

#print(type(images))
#print(images.shape)

#predictions = model(images)

#print(predictions.keys())

#sys.exit(0)
 
# 5. Train!
model.fit(train_ds, epochs=EPOCHS)
 
# 6. Save your new custom model
model.save("custom_tree_detector.keras")
