#!./tensorflow/bin/python

# Step 1: Import the all necessary libraries and SDK commands.
import os
import boto3
import botocore
import sys
import json

PROGRAM_NAME=str(sys.argv[0].lstrip('.').lstrip('/'))

if len(sys.argv) < 2:
    print(PROGRAM_NAME + ": Error: No file provided: Please provide name of file in input.", file=sys.stderr)
    sys.exit(1)

if len(sys.argv) < 3:
    print(PROGRAM_NAME + "Warning: No destination file name provided. Defaulting to path as file name.", file=sys.stderr)
    answer = input("Do you want to continue with path name?:[y/n] ")
    if (answer != "y" and answer != "Y"):
        sys.exit(1)

with open("password.json", "r") as file:
    data = json.load(file)

    # Step 2: The new session validates your request and directs it to your Space's specified endpoint using the AWS SDK.
    session = boto3.session.Session()
    client = session.client('s3',
                            region_name='sfo3', # Use the region in your endpoint.
                            endpoint_url='https://sfo3.digitaloceanspaces.com', # Find your endpoint in the control panel, under Settings. Prepend "https://".
                            aws_access_key_id=data['aws_access_key_id'], # Access key pair. You can create access key pairs using the control panel or API.
                            aws_secret_access_key=data['aws_secret_access_key'])

    #client.create_bucket(Bucket='tcb-drone')

    #print(str(client))
    # List all buckets on your account.
    response = client.list_buckets()
    spaces = [space['Name'] for space in response['Buckets']]
    print(str(spaces))
    
    print(PROGRAM_NAME + " Info: Uploading: " + str(sys.argv[1]), file=sys.stderr)
    
    # Upload the desired file 
    
    # Step 3: Call the put_object command and specify the file to upload. 
    if len(sys.argv) < 3:
        client.upload_file(str(sys.argv[1]), 'tcb-drone', str(sys.argv[1]), ExtraArgs={'ACL':'public-read'})
    else:
        client.upload_file(str(sys.argv[1]), 'tcb-drone', str(sys.argv[2]), ExtraArgs={'ACL':'public-read'})
        
    # Get the list of files 
    
    response = client.list_objects_v2(
        Bucket='tcb-drone',
        Delimiter=' ',
        EncodingType='url'
    )
    
    fileList = []
    for entry in response['Contents']:
        fileList.append(entry['Key'])
    
    # Create an HTML file to list the files 
    
    try:             
        with open("index.html", "w") as f:
            try:
                f.write("<!DOCTYPE html>")
                f.write("<html>")
                f.write("<body>")
                f.write("<h1>File Listing</h1>")
                for entry in fileList:
                    f.write("<p><a href=\"https://tcb-drone.sfo3.digitaloceanspaces.com/" + str(entry) + "\">https://tcb-drone.sfo3.digitaloceanspaces.com/" + str(entry) + "</a></p>")

                f.write("</body>")
                f.write("</html>")
            except IOError as e:
                print(PROGRAM_NAME + ": Error: writing to file: " + str(e), file=sys.stderr)
                sys.exit(1)
            except OSError as e:
                print(PROGRAM_NAME + ": Error: writing to file: " + str(e), file=sys.stderr)
                sys.exit(1)
    except FileNotFoundError as e:
        print(PROGRAM_NAME + ": Error: opening file: " + str(e), file=sys.stderr)
        sys.exit(1)
    except PermissionError as e:
        print(PROGRAM_NAME + ": Error: opening file: " + str(e), file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(PROGRAM_NAME + ": Error: opening file: " + str(e), file=sys.stderr)
        sys.exit(1)
        
    # Upload the updated html file 
    
    client.upload_file(str("index.html"), 'tcb-drone', str("index.html"), ExtraArgs={'ACL':'public-read'})
    sys.exit(0)

