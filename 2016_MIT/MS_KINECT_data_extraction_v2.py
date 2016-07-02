'''
Dependency : freenect + python-wrap + opencv + opengl
'''

from freenect import sync_get_depth as get_depth, sync_get_video as get_video
import numpy as np
import imageio
from skimage import io
import cv2
import argparse
import os
from datetime import datetime

# Set Variables
frame_length_limit_order=6
__author__ = 'Kent (Jin-Chun Chiu)'


def get_args():
    '''This function parses and return arguments passed in'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description=''' Script read both depth and rgb frames from kinetic 
        and save them in Assigned Folder''')
    # Add arguments
    parser.add_argument(
        '-fn', '--folder_name', type=str, help='Object or Event as Folder name', required=True)
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    folder_name = args.folder_name
    # Return all variable values
    return folder_name

def depth_to_gray(depth):
    depth = depth*(255.00/2047.00)
    depth = depth.astype(int)
    return depth

def main():
    print ('[*] Start')
    start = datetime.now()
    folder_name=get_args()
    os.mkdir(folder_name)
    os.chdir(folder_name)
    i=1
    try:
        while True:
            (depth,_), (rgb,_) = get_depth(), get_video()
            a=""
            if len(str(i))<frame_length_limit_order:
                a='0'*(frame_length_limit_order-len(str(i)))
            elif len(str(i))>frame_length_limit_order:
                break
            depth=depth_to_gray(depth)
            io.imsave('depth'+a+str(i)+'.png',depth)
            io.imsave('rgb'+a+str(i)+'.png',rgb)
            i+=1
    except KeyboardInterrupt:
        duration=str((datetime.now()-start).total_seconds() ).split('.')[0]
        print '\n[*] Recoding Duration is { %s }'%(duration)

if __name__=='__main__':
    main()