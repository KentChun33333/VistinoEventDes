import numpy as np
import argparse
import imutils
import cv2
import motion_interpretation
from scipy.stats.mstats import mode as statics_mode

from HUB_dictionary.motion_dictionary  import Motion_Dictionary
from HUB_dictionary.path_dictionary    import Path_DTW_Dictionary
from HUB_dictionary.gesture_dictionary import Gestrue_DTW_Dictionary

import time

class Recog2Track():
	def __init__(self, conf):
		self.numFrames = 0
		self.gesture = None
		self.values = []
		self.motionALL = []
		self.motionSeq = []
		self.motionLikehoodSeq = []
		self.Dictionary = Motion_Dictionary()
		self.PathDictionary = Path_DTW_Dictionary()
		self.PathRegStr = "None" 

# Get the varoable from command line
folder_name=get_args()

camera = cv2.VideoCapture(0)

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