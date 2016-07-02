'''
Dependency : freenect + python-wrap + opencv + opengl
Description : This is for MS_Kinectic_V1
'''

from freenect import sync_get_depth as get_depth, sync_get_video as get_video, sync_stop
import numpy as np
import imageio
from skimage import io
import cv2
import argparse
import os
from datetime import datetime
import time

# Set Variables
frame_length_limit_order=6
__author__ = 'Kent (Jin-Chun Chiu)'

# Def 
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
    buff=[]
    try:
        while True:
            print '[*] Recording Index %s'%(str(i))
            (depth,_), (rgb,_) = get_depth(), get_video()
            buff.append([depth.copy(),rgb.copy()])
            i+=1
            time.sleep(0.0001)
            if i>=1000000:
                break
    except KeyboardInterrupt:
        sync_stop() # stop the sync_get_video...etc
        print '\n[*] End Buff with following information :'
        duration = str((datetime.now()-start).total_seconds() ).split('.')[0]
        fps = i/((datetime.now()-start).total_seconds())
        print '[*] Duration is { %s }'%(duration)
        print '[*] FPS is { %s }'%(str(fps).split('.')[0])

        print '\n[*] Start Saving IMG from Buff'
        try:
            for j in range(i):
                #list.pop(index)
                depth,rgb = buff.pop(0)
                a=""
                if len(str(j))<frame_length_limit_order:
                    a='0'*(frame_length_limit_order-len(str(j)))
                depth=depth_to_gray(depth)
                # io.imsave to a series of .png
                io.imsave('depth'+a+str(j)+'.png',depth)
                io.imsave('rgb'+a+str(j)+'.png',rgb)
                print '[*] Saving Index %s'%(str(j))
        except:
            print '\n[*] End Saving IMG '

if __name__=='__main__':
    main()
