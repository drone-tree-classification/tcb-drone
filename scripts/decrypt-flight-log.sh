#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 [path to encrypted flight log] [path to output file]"
    exit 1
fi

SCRIPT_DIR=$(dirname "${BASH_SOURCE}")
FRPL_DIR="FlightRecordParsingLib"
FRPL_BUILD_DIR="FlightRecordParsingLib/dji-flightrecord-kit/build/Ubuntu/FRSample"

pushd $SCRIPT_DIR > /dev/null

# If the dji git project is not cloned, clone it.
if [ ! -d "$FRPL_DIR" ]; then
    git clone git@github.com:dji-sdk/FlightRecordParsingLib.git
fi

# If the sample decoder is not built, build it.
if [ ! -f "${FRPL_BUILD_DIR}/FRSample" ]; then
    pushd "${FRPL_BUILD_DIR}"  > /dev/null
    sh ./generate.sh
    popd  > /dev/null
fi

# Check if API key is defined in env or file
if [ -z "$SDK_KEY" ]; then
    if [ -f ~/.dji_key ]; then
        export SDK_KEY=$(cat ~/.dji_key)
    else
        echo "ERROR: DJI key must be defined in environment cariable SDK_KEY or file ~/.dji_key"
        exit 1
    fi
fi

# Run the sample decoder on our input file
${FRPL_BUILD_DIR}/FRSample $1 > $2

popd  > /dev/null