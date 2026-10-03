#!/usr/bin/env bash

# This script sets up the environment variables needed to run the
# tcb-drone scripts easily from the root directory.
#
# Usage: source setup-environment.sh

# Get the absolute path of the directory containing this script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Add the scripts directory to the PATH
export PATH="${SCRIPT_DIR}/scripts:${PATH}"

# Add the libs directory to PYTHONPATH so scripts can resolve the modules
export PYTHONPATH="${SCRIPT_DIR}/libs:${PYTHONPATH}"

TRAINING_SET=("RillitoPark" "CherryPark")
DOWNLOADS_DIR="${SCRIPT_DIR}/Downloads"

for t in "${TRAINING_SET[@]}"; do
    if [ ! -f "${DOWNLOADS_DIR}/${t}" ] && [ ! -d "${DOWNLOADS_DIR}/${t}" ]; then
        echo "Downloading training set ${t} into Downloads/ ..."
        wget -q -P "${DOWNLOADS_DIR}" "https://tcb-drone.sfo3.digitaloceanspaces.com/${t}"

        # Extract the training set unconditionally, like the original script did
        echo "Extracting ${t}..."
        tar -xf "${DOWNLOADS_DIR}/${t}" -C "${DOWNLOADS_DIR}"
    else
        echo "Training set ${t} is already in Downloads/."
    fi
done

echo "Environment initialized. Scripts in ./scripts are now available in your PATH."
