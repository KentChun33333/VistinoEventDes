import cv2
import imageio
import os
from skimage import io 
import argparse

#
def get_args():
    '''This function parses and return arguments passed in'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description=''' Script read both depth and rgb frames from kinetic 
        and save them in Assigned Folder''')
    # Add arguments
    parser.add_argument(
        '-fn', '--folder_name', type=str, help='Object or Event as Folder name', required=True)
    ###parser.add_argument(
    ###   '-p', '--port', type=str, help='Port number', required=True, nargs='+')
    ###parser.add_argument(
    ###    '-k', '--keyword', type=str, help='Keyword search', required=False, default=None)
    # Array for all arguments passed to script
    args = parser.parse_args()

    # Assign args to variables
    folder_name = args.folder_name
    ###port = args.port[0].split(",")
    ###keyword = args.keyword
    # Return all variable values
    return folder_name


methods = [
	("cv2.INTER_NEAREST", cv2.INTER_NEAREST),
	("cv2.INTER_LINEAR", cv2.INTER_LINEAR),
	("cv2.INTER_AREA", cv2.INTER_AREA),
	("cv2.INTER_CUBIC", cv2.INTER_CUBIC),
	("cv2.INTER_LANCZOS4", cv2.INTER_LANCZOS4)]


def main():
    resized = cv2.resize(img1, (width,height), interpolation=cv2.INTER_CUBIC)
