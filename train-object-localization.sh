#!/usr/bin/bash

# train-object-localization is a reproducable training script for the object
# localization file creation. Executing this file will download the training
# set, train the object localization model, create a .keras file, and ask the 
# user whether they would like to upload the file to the digital ocean bucket. 

readonly SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Have to use the python version that is pointed to by the symbolic link, cannot resolve to the base python version
readonly PYTHON_INTERPRETER=${SCRIPT_DIR}/tensorflow/bin/python
readonly HAS_PYTHON=$(ls ${PYTHON_INTERPRETER} 2> /dev/null | wc -l) 

# Test to see if users computer has python installed 
if [ ${HAS_PYTHON} -eq 0 ]; then 
    >&2 printf "${0}: Error: python not found.\n"
    >&2 printf "${0}: Info: Run ./initialize-enviroment.sh and try again.\n"
    exit 1
fi

DATE=$(date '+%Y-%m-%d-%H-%M-%S')
readonly LATEST_KERAS_FILE=FineTunedResNetOD-${DATE}.keras

USER_RESPONSE='n'

# Could not find 
# - CSMGummyDrone 
# in ./list-files-in-space.py
declare -a TRAINING_SET=(
                           "CSMGummy2.tar.gz"
                           "Himmel.tar.gz"
                           "Himmel2.tar.gz"
                           "CherryAvePark.tar.gz"
                           "RillitoPark.tar.gz"
                )
   
# Training set dir names maps the tar files to the names that they come out of
# storage as.              
declare -a TRAINING_SET_DIR_NAMES=(
                           "CSMGummy2"
                           "HimmelDrone"
                           "HimmelDrone2"
                           "CherryAvePark"
                           "RillitoPark"
                )

# Training set string contains the linearized list of directories 
TRAINING_SET_STRING=""

# Download the training set if the file does not exist 
j=0
for i in "${TRAINING_SET[@]}"
do
    DIRNAME=${TRAINING_SET_DIR_NAMES[$j]}
    TRAINING_SET_STRING="${TRAINING_SET_STRING} ${DIRNAME}"
    if [ ! -d "${DIRNAME}" ]; then
        >&2 printf "${0}: Info: Downloading %s ...\n" ${i}
        # Download the file and untar it 
        ${SCRIPT_DIR}/download-file-from-space.sh ${i}
        ERROR=$?

        if [ ${ERROR} -ne 0 ]; then 
            >&2 printf "${0}: Error: Could not download file: %s\n" ${i}
            exit 1
        fi
        
        tar -xzvf ${i}
    fi
    j=$((j += 1))
done

echo ${TRAINING_SET_STRING}

# Train the model 
time ${PYTHON_INTERPRETER} ${SCRIPT_DIR}/tensor-train-transfer.py -o ${LATEST_KERAS_FILE} -d ${TRAINING_SET_STRING} 
PYTHON_ERROR=$?

# If training was successful, ask the user if they would like to upload the .keras file generated from training
if [ ${PYTHON_ERROR} -eq 0 ]; then
    >&2 printf "${0}: Query: Training successful, would you like to upload the model? (y/N): "
    read USER_RESPONSE
    if [ ! "${USER_RESPONSE}" == "y" ]; then
        ${PYTHON_INTERPRETER} ${SCRIPT_DIR}/upload-file-to-space.py ${LATEST_KERAS_FILE} ${LATEST_KERAS_FILE} 
        PYTHON_ERROR=$?
        if [ ${PYTHON_ERROR} -ne 0 ]; then
            >&2 printf "${0}: Error: Uploading file, please review log and upload %s manually" ${LATEST_KERAS_FILE}
        fi
    fi
fi

