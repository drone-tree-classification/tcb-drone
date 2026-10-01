#!/usr/bin/env bash

# This script sets up the environment variables needed to run the
# tcb-drone scripts easily from the root directory.
#
# Usage: source setup-environment.sh

# Get the absolute path of the directory containing this script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Add the scripts directory to the PATH
export PATH="${SCRIPT_DIR}/scripts:${PATH}"

# Add the root directory to PYTHONPATH so scripts can resolve the 'libs' module
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

echo "Environment initialized. Scripts in ./scripts are now available in your PATH."
