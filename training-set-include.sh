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

download_training_set() {
    SCRIPT_DIR_DOWNLOAD=$1
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
            ${SCRIPT_DIR_DOWNLOAD}/download-file-from-space.sh ${i}
            ERROR=$?

            if [ ${ERROR} -ne 0 ]; then 
                >&2 printf "${0}: Error: Could not download file: %s\n" ${i}
                exit 1
            fi
        
            tar -xzvf ${i}
        fi
        j=$((j += 1))
    done
    printf "%s\n" "${TRAINING_SET_STRING}"
}

