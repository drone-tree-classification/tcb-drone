# training-set-include.sh contains the list of training set files and a bash
# script to download the training set. 
# This file was created to be a single point of control because the scripts
# initialize-enviroment.sh and train-object-localization.sh both download
# the training set from the digital ocean bucket. 

declare -a TRAINING_SET=(
                           "CSMGummy.tar.gz"
                           "CSMGummy2.tar.gz"
                           "Himmel.tar.gz"
                           "Himmel2.tar.gz"
                           "CherryAvePark.tar.gz"
                           "RillitoPark.tar.gz"
                )
   
# Training set dir names maps the tar files to the names that they come out of
# storage as.              
declare -a TRAINING_SET_DIR_NAMES=(
                           "CSMGummyDrone"
                           "CSMGummy2"
                           "HimmelDrone"
                           "HimmelDrone2"
                           "CherryAvePark"
                           "RillitoPark"
                )

# download_training_set downloads the full training set
# @param PARENT_SCRIPT_NAME  name of calling script, used for error reporting, 
#  can be any string. 
# @param SCRIPT_DIR_DOWNLOAD  path to scripts directory.
# @param DOWNLOAD_DIR  directory to download to.
#
download_training_set() {
    PARENT_SCRIPT_NAME=$1
    SCRIPT_DIR_DOWNLOAD=$2
    DOWNLOAD_DIR=$3
    
    if [ $# -ne 3 ]; then
        echo "Usage: $0 [parent script name] [script dir] [download dir]"
        exit 1
    fi
    
    TRAINING_SET_STRING=""
    
    # Download the training set if the file does not exist 
    j=0
    for i in "${TRAINING_SET[@]}"
    do
        DIRNAME=${TRAINING_SET_DIR_NAMES[$j]}
        TRAINING_SET_STRING="${TRAINING_SET_STRING} ${DIRNAME}"
        if [ ! -d "${DIRNAME}" ]; then
            >&2 printf "%s: Info: Downloading %s ...\n" ${PARENT_SCRIPT_NAME} ${i}
            # Download the file and untar it 
            ${SCRIPT_DIR_DOWNLOAD}/download-file-from-space.sh ${i}
            ERROR=$?

            if [ ${ERROR} -ne 0 ]; then 
                >&2 printf "%s: Warning: Could not download file: %s\n" ${PARENT_SCRIPT_NAME} ${i}
                continue
            fi
            
            mv ${i} ${DOWNLOAD_DIR}
            ERROR=$?

            if [ ${ERROR} -ne 0 ]; then 
                >&2 printf "%s: Warning: Could not move file to: %s\n" ${PARENT_SCRIPT_NAME} ${DOWNLOAD_DIR}
                continue
            fi
            
            tar -xzvf ${DOWNLOAD_DIR}"/"${i}
            
            ERROR=$?
            if [ ${ERROR} -ne 0 ]; then 
                >&2 printf "%s: Warning: Could not untar file: %s\n" ${PARENT_SCRIPT_NAME} ${DOWNLOAD_DIR}"/"${i}
                continue
            fi
        fi
        j=$((j += 1))
    done
    printf "%s\n" "${TRAINING_SET_STRING}"
}

