#def run(filename):
#	'''Method that executes the video sequence'''
#	global processedImg
#	cap = cv2.VideoCapture(filename)
#	setupSliders()
#	_,f = cap.read()
#	currentImg = f
#	img = tr.handleFrame(f)
#	h, w, d = img.shape
#	video = cv2.VideoWriter("tester.avi", cv.CV_FOURCC('M','J','P','G'), 10.0,(w, h),True) #Make a video writer
#	cv2.imshow('Output',img)
#	while(True):
#		vals = getSliderValues()
#		if vals["start"] == 1:
#			isOk,f = cap.read()
#			if isOk:
#				currentImg = f
#				img = tr.handleFrame(f, vals)
#				video.write(img)
#				cv2.imshow('Output',img)
#		if cv2.waitKey(5)==27:
#			break
#		cmd = cv2.waitKey(5)
#		if cmd==ord('r'):
#			tr.startRecording()
#	video.release()
#	showPupilAreas(tr.getPupilAreas())
#	cv2.destroyAllWindows()


	#####
	#####
	#!/usr/local/bin/python3

import cv2
import argparse
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
args = vars(ap.parse_args())

# Arguments
dir_path = '.'
ext = args['extension']
output = args['output']

images = []
for f in os.listdir(dir_path):
    if f.endswith(ext):
        images.append(f)

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video',frame)

height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 1.0, (width, height))

for image in images:

    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)

    out.write(frame) # Write out frame to video

    cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print ("The output video is {}".format(output))


#tk-img2video -ext png -o output.mp4
