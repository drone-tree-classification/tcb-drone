#/bin/bash

# This script sets up your TCB-Drone project for TensorFlow development. 
# This script is written for Linux and assumes that you already have the 
# python3-venv package installed. 
# If the package is not installed, you may install it with the command 
# sudo apt install python3-venv 

# This script should be run just once, after the repo has been cloned and 
# before any python scripts are run. 

# Name of the training set file, this may change PR to PR. 
TRAINING_SET=(Parker.tar.gz RillitoPark CherryPark)

# Run with GPU arg to set up tensorflow with cuda. To download cuda toolkit, reference:
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux

# Verify installation 
IS_INSTALLED=0
tf_ver=""

readonly HAS_PYTHON=$(which python3 | wc -l)

if [ ${HAS_PYTHON} -eq 0 ]; then 
    >&2 printf "${0}: Error: python not found.\n"
    >&2 printf "${0}: Info: please install python3 and try again.\n"
    exit 1
fi

PYTHON_MAJOR_NEEDED=3
PYTHON_MINOR_MIN=10

PYTHON_MAJOR=$(python3 --version | grep -Po '[0-9]+\.[0-9]+' | awk '{split($1,a,".");print a[1]}')
PYTHON_MINOR=$(python3 --version | grep -Po '[0-9]+\.[0-9]+' | awk '{split($1,a,".");print a[2]}')

if [ ${PYTHON_MAJOR} -ne ${PYTHON_MAJOR_NEEDED} ]; then 
    >&2 printf "${0}: Error: python version needs to be %d, your version is: %d.\n" ${PYTHON_MAJOR_NEEDED} ${PYTHON_MAJOR}
    >&2 printf "${0}: Info: please install python%d and try again.\n" ${PYTHON_MAJOR_NEEDED}
    exit 1
fi

if [ ${PYTHON_MINOR} -lt ${PYTHON_MINOR_MIN} ]; then 
    >&2 printf "${0}: Error: python version needs to be 3.%d, or higher, your version is: 3.%d.\n" ${PYTHON_MINOR_MIN} ${PYTHON_MINOR}
    >&2 printf "${0}: Info: please install python3.%d and try again.\n" ${PYTHON_MINOR_MIN}
    exit 1
fi

if [[ $1 == "GPU" ]]; then
    tf_ver="tensorflow[and-cuda]"
    check_tf_installed () {
        # Check that a GPU device is listed
        GPU_DEVICE=$(python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" | grep -Po "\[PhysicalDevice\(.*\)\]")
        echo "GPU Device Found: ${GPU_DEVICE}"
        if [[ -n "${GPU_DEVICE}" ]]; then
            IS_INSTALLED=1
        fi
    }
else
    tf_ver="tensorflow"
    check_tf_installed () {
       # Verify installation 
        IS_INSTALLED=$(python3 -c "import tensorflow as tf; print(tf.__version__)"  | grep -Po "[0-9]+\.[0-9]+\.[0-9]+"  | wc -l)
    }
fi

if [ -f tensorflow/bin/activate  ]; then
    source tensorflow/bin/activate
    check_tf_installed
fi

# If not already installed, get tensor flow 
if [[ ${IS_INSTALLED} -eq 0 ]]; then
    # Delete the tensorflow folder if one exists and is not a verified install 
    rm -rf tensorflow/
    # Get tensorflow (CPU Version) 
    python3 -m venv tensorflow 
    source tensorflow/bin/activate 
    pip install --upgrade pip 
    pip install --upgrade ${tf_ver}
    pip install opencv-python
    pip install pyyaml h5py  # Required to save models in HDF5 format
    pip install botocore 
    pip install boto3
    pip install matplotlib
    
    # Verify installation 
    check_tf_installed

    if [[ ${IS_INSTALLED} -gt 0 ]]; then
        echo "Installation Successful"
    else
        echo "Installation unsuccessful"
        exit 1
    fi
fi

# If the training set is not already downloaded, download it  
for t in ${TRAINING_SET[@]}; do
    if [ ! -f ${t} ] && [ ! -d ${t} ]; then
        # Get the initial training set 
        wget https://tcb-drone.sfo3.digitaloceanspaces.com/${t}

        # unzip the training set 
        tar -xf ${t}
    else
        echo "${0}: info: ${t} Already Downloaded."
    fi
    
done



