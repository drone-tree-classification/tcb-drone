#!/usr/bin/env bash

# This script runs all unit tests in the unit test directory 

# Set up the PYTHON PATH to enable import statements in unit tests to work 
export PYTHONPATH=${PYTHONPATH}:`pwd`/../libs/:`pwd`/../:`pwd`/../scripts/

RETURN=0

# Search for all python files, run and execute them 
for entry in $(find . -type f -name "*.py")
do
  python3 $entry -v
  ERROR=$?
  if [ ${ERROR} -ne 0 ]; then 
    RETURN=${ERROR}
  fi
done

exit ${RETURN}
