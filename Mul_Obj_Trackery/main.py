# USAGE

# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import motion_interpretation
from scipy.stats.mstats import mode as statics_mode
from motion_dictionary  import Motion_Dictionary
from path_dictionary    import Path_DTW_Dictionary
from gesture_dictionary import Gestrue_DTW_Dictionary
import fastdtw
import time

def get_args():
    '''This function parses and return arguments passed in'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description=''' Script read both depth and rgb frames from kinetic 
        and save them in Assigned Folder''')
    # Add arguments
    parser.add_argument(
        '-fn', '--folder_name', type=str, help='file name.txt', required=True)
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    folder_name = args.folder_name
    # Return all variable values
    return folder_name


def sequence_container(inPut, seqLen):
    if len(inPut)<seqLen:
        return inPut
    return inPut[-seqLen:]

def red_finder(img):
    lowerBound = np.array([0,0,100])
    upperBound = np.array([62,62,255])	
    # [*] Use RGB model
    mask = cv2.inRange(img, lowerBound, upperBound)
    # [*] Mask already be binary_picture ..... 

    # [*] Turn Gray 
    gray = cv2.cvtColor(mask, cv2.COLOR_BAYER_GB2GRAY)
    blurred=cv2.blur(gray,(5,5))
    # Set 200, more than 200 ---> asain to 255
    (_, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # [*] MORPH_OPEN - an opening operation
    # [*] MORPH_CLOSE - a closing operation
    # [*] MORPH_GRADIENT - a morphological gradient
    # [*] MORPH_TOPHAT - 'top hat'
    # [*] MORPH_BLACKHAT - 'black hat'
    # [*] MORPH_HITMISS - 'hit and miss'
    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)
    return closed

def find_coutur_record(img):
	(_, cnts, _) = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	# 
	clone = img.copy()
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
	cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)
	cx = 0
	cy = 0
	if len(cnts)>0 :
		M = cv2.moments(cnts[-1])
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
	return clone,(cx,cy)

def from_cnt_cental(cnts):
	M = cv2.moments(cnts)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	return (cx,cy)



# Get the varoable from command line
folder_name=get_args()

camera = cv2.VideoCapture(0)

# initialize the total number of frames read thus far, a bookkeeping variable used to
# keep track of the number of consecutive frames a gesture has appeared in, along
# with the values recognized by the gesture detector
numFrames = 0
gesture = None
values = []
motionALL = []
motionSeq = []
motionLikehoodSeq = []
Dictionary = Motion_Dictionary()
PathDictionary = Path_DTW_Dictionary()
#init 
PathRegStr = "None" 
# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()
	# resize the frame and flip it so the frame is no longer a mirror view
	frame = imutils.resize(frame, width=600)
	frame = cv2.flip(frame, 1)
	clone = frame.copy()
	(frameH, frameW) = frame.shape[:2]

	# extract the ROI, passing in right:left since the image is mirrored, then
	# blur it slightly
	roi = frame#[top:bot, right:left]
	#-------------------------------------#
	# Recognition or Tracking or Matching #
	#-------------------------------------#

	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)

	img_red = red_finder(frame)
	img_red_check, point = find_coutur_record(img_red)

	#------------------------#
	# After point extraction #
	#------------------------#
	# print point
	if point[0] != 0 and point[1] != 0:
	# process motion.interpreter
		if len(values)>2:
			last_point = values[-1][1]
			tmp_motion = motion_interpretation.interpreter(last_point,point)

			# 
			motionALL.append(tmp_motion) # this is all motion
			motionSeq.append(tmp_motion) # this is time-segment motion
			# Compute the max_Likelihood
			motionSeq      = sequence_container(motionSeq,10)
			motionLikehood = statics_mode(motionSeq)[0]
			motionLikehoodSeq.append(motionLikehood)
			
			motionStr      = Dictionary.check(motionLikehood)

			print motionStr
			# %.2f = float with 2 dicimal 
			cv2.putText(img_red_check, "The Hand is : %s" % (motionStr), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3)
			if len(motionLikehoodSeq)>10:
				# put it in small LSTM or RNN for NPL
				# put it into DTW search/match
				motionLikehoodSeq = sequence_container(motionLikehoodSeq,10)
				PathRegStr_tmp = PathDictionary.search(motionLikehoodSeq[-8:])
				if PathRegStr_tmp == 'None':
					pass
				else:
					PathRegStr = PathRegStr_tmp
				pass # do some 3rd Recognition

		# update the whole_value
		values.append([numFrames,point])
		
		# [*] show the frame to our screen
	##cv2.imshow("Frame", clone)
	##cv2.imshow("red_finder", img_red)
	cv2.putText(img_red_check, "Last Path-Recog. : %s" % (PathRegStr), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3)
	cv2.imshow("Cou", img_red_check)
	##cv2.imshow("CheckHSV", img_check)

	numFrames += 1
	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

with open(folder_name, 'w') as f:
    for s in values:
    	s = str(s)
        f.write(s + '\n')

#with open(the_filename, 'r') as f:
#    my_list = [line.rstrip('\n') for line in f]

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

import pandas as pd

