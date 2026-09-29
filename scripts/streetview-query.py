#!/usr/bin/env python3

# Script for getting random Tucson, AZ streetview images to tag and train tree detection on
# Useful docs:
# https://medium.com/@seelcs12/querying-google-streetview-images-with-python-f486910d0345
# https://developers.google.com/maps/documentation/streetview/request-streetview

import requests
import os
import random
import argparse
import math

MAX_QUERIES=10000
OUTDIR="streetview-images"
TUCSON_DATA={"latitude": 32.232540,"longitude": -110.916055}
RADIUS=3 #Mile radius around coordinate to query

# Latitude/longitude range based on given radius and starting latitude
LATITUDE_PLUS_MINUS = RADIUS/69
LONGITUDE_PLUS_MINUS = RADIUS/(69*math.cos(TUCSON_DATA["latitude"]*math.pi/180))

def query_random_location(google_api_key):

    pic_base = 'https://maps.googleapis.com/maps/api/streetview?'

    random_lat = random.uniform(-1.0,1.0) * LATITUDE_PLUS_MINUS
    random_long = random.uniform(-1.0,1.0) * LONGITUDE_PLUS_MINUS

    # Calculating new latitude and longitude values
    new_lat, new_long = round(TUCSON_DATA["latitude"] + random_lat,6), round(TUCSON_DATA["longitude"] + random_long,6)

    # Define the params for the picture request
    pic_params = {'key': google_api_key, 'location': f"{new_lat},{new_long}", 'size': "500x500", 'return_error_code':'true'}

    #Requesting data
    pic_response = requests.get(pic_base, params=pic_params)
    if pic_response.status_code == 404:
        print(f"No image found for {new_lat},{new_long}")
    else:
        image_name = f"{new_lat}_{new_long}.jpg"
        with open( os.path.join(OUTDIR,image_name), "wb") as file:
            file.write(pic_response.content)

    # Closing connection to API
    pic_response.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='StreetViewQuery', description='Get random Tucson, AZ streetview images')

    parser.add_argument('-a','--api_key_file',
                        default='~/.google-api-key',
                        help='File containing google cloud api key')
    parser.add_argument('-n','--numqueries',
                        type=int,
                        default=5,
                        help='Number of images to return')

    args=parser.parse_args()

    random.seed(4)
    google_api_key = open( os.path.expanduser(args.api_key_file) , "r").read()
    totalqueries = open('numqueries.txt', "r").read()
    totalqueries = int(totalqueries)

    if not os.path.isdir(OUTDIR):
        os.makedirs(OUTDIR)

    try:
        for i in range(0,args.numqueries):
            if totalqueries < MAX_QUERIES:
                query_random_location(google_api_key)
                totalqueries += 1
            else:
                print(f"Reached max query limit of {MAX_QUERIES}, exiting...")
                break
    except Exception as e:
        print("Caught an error, writing query number and exiting")
        print(e)

    with open('numqueries.txt', "w") as qfile:
        qfile.write(str(totalqueries))

    print("Done")