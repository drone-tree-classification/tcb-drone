#!./tensorflow/bin/python

# Step 1: Import the all necessary libraries and SDK commands.
import os
import boto3
import botocore
import sys
import json
import datetime
import argparse

def matches_extension(ext_list: str, filename: str) -> bool:
    """
    ext_list: comma-separated extensions, e.g. ".jpg,.jpeg,.JPG"
    filename: file name to test, e.g. "file.jpg"
    """
    if not filename or "." not in filename:
        return False

    extensions = {ext.strip() for ext in ext_list.split(",")}

    return any(filename.endswith(ext) for ext in extensions)

def human_readable_bytes(nbytes):
    """
    Converts a byte count into a human-readable string with appropriate suffixes.

    Args:
        nbytes (int): The number of bytes.

    Returns:
        str: A string representing the human-readable byte size.
    """
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
    i = 0
    while nbytes >= 1024 and i < len(suffixes) - 1:
        nbytes /= 1024.0
        i += 1
    
    # Format the number to two decimal places, removing trailing zeros and the decimal point if not needed
    formatted_nbytes = f'{nbytes:.2f}'.rstrip('0').rstrip('.')
    return f'{formatted_nbytes} {suffixes[i]}'

if __name__ == "__main__":
    PROGRAM_NAME=str(sys.argv[0].lstrip('.').lstrip('/'))

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest='name_only', action='store_true')
    parser.add_argument('-e', dest='extensions', default="")
    args = parser.parse_args()
    name_only=args.name_only
    extensions=args.extensions

    try:
        with open("password.json", "r") as file:
            data = json.load(file)

            # Step 2: The new session validates your request and directs it to your Space's specified endpoint using the AWS SDK.
            session = boto3.session.Session()
            client = session.client('s3',
                                    region_name='sfo3', # Use the region in your endpoint.
                                    endpoint_url='https://sfo3.digitaloceanspaces.com', # Find your endpoint in the control panel, under Settings. Prepend "https://".
                                    aws_access_key_id=data['aws_access_key_id'], # Access key pair. You can create access key pairs using the control panel or API.
                                    aws_secret_access_key=data['aws_secret_access_key'])

            # List all buckets on your account.
            try:
                # Get the list of files 
                response = client.list_objects_v2(
                    Bucket='tcb-drone',
                    Delimiter=' ',
                    EncodingType='url'
                )
                
                fileKeyValuePairList = []
                
                maxlen = 0
                for entry in response['Contents']:
                    tempStr = str(entry['Key']).replace('%2F', '/')
                    fileKeyValuePairList.append(
                        {
                            "Key": tempStr
                            ,"Size": entry['Size']
                        }
                    )
                    if maxlen < len(str(tempStr)):
                        maxlen = len(str(tempStr))
                
                # We want to sort with respect to time 
                sorted_by_name_desc = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=False)
                
                # https://tcb-drone.sfo3.digitaloceanspaces.com/" + 
                
                print("File path:".ljust(maxlen + 1) + " Size:", file=sys.stderr)
                print("----------".ljust(maxlen + 1) + " -----", file=sys.stderr)
                for entry in fileKeyValuePairList:
                    if matches_extension(extensions, str(entry['Key'])):
                        if name_only:
                            print((str(entry['Key'])).ljust(maxlen + 1))
                        else:
                            print((str(entry['Key'])).ljust(maxlen + 1) + " " + str(human_readable_bytes(entry['Size'])))
                
                sys.exit(0)
                
            except botocore.exceptions.ClientError as error:
                print(PROGRAM_NAME + ": Error: " + str(error) + "", file=sys.stderr)
                sys.exit(2)

            except botocore.exceptions.ParamValidationError as error:
                print(PROGRAM_NAME + ": Error: " + str(error) + "", file=sys.stderr)
                sys.exit(3)

    except FileNotFoundError:
        print(PROGRAM_NAME + ": Error: password.json not found.", file=sys.stderr)
        sys.exit(1)
    
