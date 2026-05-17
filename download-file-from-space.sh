#!/usr/bin/bash

# download-file-from-space.sh attempts to download the file or directory 
# specified by the path on the Digital Ocean bucket. 
# Usage: ./download-file-from-space.sh file-name [-f]
# -f flag is for force. It will allow files to be overwritten. 

readonly URL_PREFIX="https://tcb-drone.sfo3.digitaloceanspaces.com/"
FORCE=0
USER_RESPONSE='y'
readonly SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
readonly PYTHON_INTERPRETER=$(realpath ${SCRIPT_DIR}/tensorflow/bin/python)
# Check for the presense of utilities 
readonly HAS_PERROR=$(which perror | wc -l)
readonly HAS_WGET=$(which wget | wc -l)
readonly HAS_GREP=$(which grep | wc -l)
readonly HAS_TAIL=$(which tail | wc -l)
readonly HAS_PYTHON=$(ls ${PYTHON_INTERPRETER} 2> /dev/null | wc -l) 

if [ ${HAS_WGET} -eq 0 ]; then 
    >&2 printf "${0}: Error: wget is required, but not installed, please install wget.\n"
    exit 1
fi

if [ $# -lt 1 ] || [ "$1" == "-f" ]
then 
    >&2 printf "${0}: Error: No file specified. Please provide name of file to download.\n"
    exit 1
fi

# Check for presense of force flag 
if [ $# -ge 2 ] && [ "$2" == "-f" ] 
then
    FORCE=1
fi

# If the file exists and the force flag is not set, ask the user if they want to continue download 
if [ -f $1 ] && [ ${FORCE} -eq 0 ] 
then
   >&2 printf "${0}: Query: File exists, continue? (y/N): "
   read USER_RESPONSE
fi

if [ ! "${USER_RESPONSE}" == "y" ]; then
    >&2 printf "${0}: Info: aborting download: ${1}\n" 
    exit 0
fi

# Attempt to download the file 
wget ${URL_PREFIX}${1} 2> /dev/null 
mv ${1} Downloads/${1}

ERROR=$?

# If download was unsuccessful
# Server issued an error response. Exec format error.
# This means that the user is trying to download a whole directory 
if [ ${ERROR} -eq 8 ]; then 

    # Check if required applications are installed    
    if [ ${HAS_TAIL} -eq 0 ]; then 
        >&2 printf "${0}: Error: tail is required, please install tail.\n"
        exit 1
    fi
    
    if [ ${HAS_GREP} -eq 0 ]; then 
        >&2 printf "${0}: Error: grep is required, please install grep.\n"
        exit 1
    fi
    
    if [ ${HAS_PYTHON} -eq 0 ]; then 
        >&2 printf "${0}: Error: python not found.\n"
        >&2 printf "${0}: Info: Run ./initialize-enviroment.sh and try again.\n"
        exit 1
    fi
    
    # See if name was a dir
    thestring=$1
    last_char=$(echo -n "$thestring" | tail -c 1)
    
    if [ ! "${last_char}" == "/" ]; then
        thestring="${thestring}/"
    fi
    
    set -o pipefail
    names=$(${PYTHON_INTERPRETER} ${SCRIPT_DIR}/list-files-in-space.py -n 2> /dev/null) 
    PYTHON_ERROR=$?
    numFilesInDir=$(echo "$names" | grep "${thestring}" | wc -l)
    if [ ${numFilesInDir} -gt 0 ]; then
        names=$(echo "$names" | grep "${thestring}")
        i=1;
        while read n; do 
          mkdir -p $(dirname ${n}) 
          if [ -f ${n} ] && [ ${FORCE} -eq 0 ] 
          then
             >&2 printf "${0}: Warning: ${n} exists, skipping. Use -f to overwrite\n"
             read USER_RESPONSE 
             if [ ! "${USER_RESPONSE}" == "y" ]; then
                 i=$(($i+1)); 
                 continue 
             fi
          fi
          >&2 printf "${0}: Info: Downloading %s: %s\n" "${URL_PREFIX}${n}" 
          wget ${URL_PREFIX}${n}
          if [ $? -ne 0 ]; then 
            if [ ${HAS_PERROR} -gt 0 ]; then 
              >&2 printf "${0}: Error: Could not download %s: %s\n" "${n}" "`perror ${ERROR}`" 
            else
              >&2 printf "${0}: Error: Could not download %s: error code: %d\n" "${n}" ${ERROR}
            fi
          else
            mv `basename ${n}`  `dirname ${n}` 
          fi
          i=$(($i+1)); 
        done <<< "$names"  
    else
        >&2 printf "${0}: Error: Could not find files in the directory.\n"
        case ${PYTHON_ERROR} in
        0) >&2 printf "${0}: Info: list-files-in-space.py indicated: Success. Issue not with list script, file requested is not on the server.\n";;
        1) >&2 printf "${0}: Info: list-files-in-space.py indicated: password.json not found.\n";;
        2) >&2 printf "${0}: Info: list-files-in-space.py indicated: boto3 ClientError.\n";;
        3) >&2 printf "${0}: Info: list-files-in-space.py indicated: boto3 ParamValidationError.\n";;
        *) >&2 printf "${0}: Info: list-files-in-space.py indicated: Unknown error: %d\n" ${PYTHON_ERROR};;
        esac
        shift
        exit 1
    fi
# If the wget failed because of some other issue besides the user trying to download a directory
elif [ ${ERROR} -ne 0 ]; then 
    if [ ${HAS_PERROR} -gt 0 ]; then 
        >&2 printf "${0}: Error: Could not download file: %s\n" "`perror ${ERROR}`"
    else
        >&2 printf "${0}: Error: Could not download file, error code: %d\n" ${ERROR}
    fi
    exit 1
fi


>&2 printf "${0}: Info: download complete for: ${1}\n" 

exit 0

