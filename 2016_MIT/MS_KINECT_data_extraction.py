'''

freenect + wreap + opencv + opengl

'''

from freenect import sync_get_depth as get_depth, sync_get_video as get_video
import numpy as np
import imageio
from skimage import io
import cv2
import argparse
import os


# Global Variables
global frame_length_limit_order=6
global __author__ = 'Kent (Jin-Chun Chiu)'


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

def depth_to_gray(depth):
    depth = depth*(255.00/2047.00)
    depth = depth.astype(int)
    return depth

def main():
	folder_name=get_args()
	os.mkdir(folder_name)
	os.chdir(folder_name)
	i=1
    while True:
        (depth,_), (rgb,_) = get_depth(), get_video()
        a=""
        if len(str(i))<frame_length_limit_order:
            a='0'*(frame_length_limit_order-len(str(i)))
        elif len(str(i))>frame_length_limit_order:
            break
        depth=depth_to_gray(depth)
        io.imsave('depth'+a+str(i)+'.png',depth`)
        io.imsave('rgb'+a+str(i)+'.png',rgb)
        i+=1
if __name__=='__main___':
    main()




def depth_to_gray(depth):
    depth = depth*(255.00/2047.00)
    depth = depth.astype(int)
    return depth

io.imsave('test.png',depth) # once you save as png, it would totally no problem for further translating



'''
Sliding window method
'''




# import the necessary packages
from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window
import argparse
import time
import cv2
 

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# load the image and define the window width and height
image = cv2.imread(args["image"])
(winW, winH) = (128, 128)

# loop over the image pyramid
for resized in pyramid(image, scale=1.5):
	# loop over the sliding window for each layer of the pyramid
	for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
 
		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
		# WINDOW
 
		# since we do not have a classifier, we'll just draw the window
		clone = resized.copy()
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		cv2.imshow("Window", clone)
		cv2.waitKey(1)
		time.sleep(0.025)

'''
How to exteact contours:
1 : pre-process
2 : findcontours
3 : drawcontour
4 : bitwise with oringinow pic, and then find the contours
'''

'''
 python sliding_window.py --image images/adrian_florida.jpg
'''

##############################################################################
'''
Saving a Video

we capture a video, process it frame-by-frame and we want to save that video.
For images, it is very simple, just use cv2.imwrite(). 

Here a little more work is required.
This time we create a VideoWriter object. 
We should specify the output file name (eg: output.avi). 
Then we should specify the FourCC code (details in next paragraph). 
Then number of frames per second (fps) and frame size should be passed. 
And last one is isColor flag. 

If it is True, encoder expect color frame, 
otherwise it works with grayscale frame.

FourCC is a 4-byte code used to specify the video codec. 
The list of available codes can be found in fourcc.org. 
It is platform dependent. Following codecs works fine for me.

In Fedora: DIVX, XVID, MJPG, X264, WMV1, WMV2. (XVID is more preferable. MJPG results in high size video. X264 gives very small size video)
In Windows: DIVX (More to be tested and added)
In OSX : (I don’t have access to OSX. Can some one fill this?)
FourCC code is passed as cv2.VideoWriter_fourcc('M','J','P','G') or cv2.VideoWriter_fourcc(*'MJPG) for MJPG.

Below code capture from a Camera, flip every frame in vertical direction and saves it.
'''
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()



'''




