#!/usr/bin/bash

# validate.sh downloads the validation set and compares outputs from the model
# to annotations in the images.

# LATEST_KERAS_FILE is the most recent uploaded keras model. This should be 
# updated in a PR when a new model is released. 
readonly LATEST_KERAS_FILE=TreeIdentifyTensorFlowModelCropped_v01.keras
readonly LATEST_CLASSES_FILE=classes-cropped_v01.txt.tmp

# VALIDATION_SET is an array of 
declare -a VALIDATION_SET=(
                           "MissionManorPark.tar.gz"
                           "HimmelDrone.tar.gz"
                           "LaCanadaVCA.tar.gz"
                           "CSMGummyDrone.tar.gz"
                )

readonly SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Have to use the python version that is pointed to by the symbolic link, cannot resolve to the base python version
readonly PYTHON_INTERPRETER=${SCRIPT_DIR}/tensorflow/bin/python
readonly HAS_PYTHON=$(ls ${PYTHON_INTERPRETER} 2> /dev/null | wc -l) 

readonly KERAS_FILE=TreeIdentifyTensorFlowModelCropped.keras
readonly CLASSES_FILE=classes-cropped.txt.tmp

if [ ${HAS_PYTHON} -eq 0 ]; then 
    >&2 printf "${0}: Error: python not found.\n"
    >&2 printf "${0}: Info: Run ./initialize-enviroment.sh and try again.\n"
    exit 1
fi

if [ ! -f ${LATEST_KERAS_FILE} ]; then
    ${SCRIPT_DIR}/download-file-from-space.sh ${LATEST_KERAS_FILE}
    ERROR=$?

    if [ ${ERROR} -ne 0 ]; then 
        >&2 printf "${0}: Error: Could not download file: %s\n" ${LATEST_KERAS_FILE}
        exit 1
    fi
fi

if [ ! -f ${LATEST_CLASSES_FILE} ]; then
    ${SCRIPT_DIR}/download-file-from-space.sh ${LATEST_CLASSES_FILE}
    ERROR=$?

    if [ ${ERROR} -ne 0 ]; then 
        >&2 printf "${0}: Error: Could not download file: %s\n" ${LATEST_CLASSES_FILE}
        exit 1
    fi
fi

# Make the symbolic link from the normally named keras file to the new
# keras file  
ln -sf ${LATEST_KERAS_FILE} ${KERAS_FILE} 
ln -sf ${LATEST_CLASSES_FILE} ${CLASSES_FILE} 

# Download the validation set if the file does not exist 
for i in "${VALIDATION_SET[@]}"
do
    DIRNAME=$(basename -s .tar.gz ${i})
    if [ ! -d "Downloads/${DIRNAME}" ]; then
        # Download the file and untar it 
        ${SCRIPT_DIR}/download-file-from-space.sh ${i}
        ERROR=$?

        if [ ${ERROR} -ne 0 ]; then 
            >&2 printf "${0}: Error: Could not download file: %s\n" ${i}
            exit 1
        fi
        
        tar -xzvf Downloads/${i}
    fi
done

ACCUMULATED_DIRS=""
# Run the validation set 
for i in "${VALIDATION_SET[@]}"
do
    DIRNAME=$(basename -s .tar.gz ${i})
    ACCUMULATED_DIRS="${ACCUMULATED_DIRS} ${DIRNAME}"
done

time ${PYTHON_INTERPRETER} ${SCRIPT_DIR}/harvest-images.py ${ACCUMULATED_DIRS} | ${PYTHON_INTERPRETER} ${SCRIPT_DIR}/tensor-run-crop.py | ${PYTHON_INTERPRETER} ${SCRIPT_DIR}/count-matches.py ${CLASSES_FILE}

