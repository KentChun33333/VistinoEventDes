
#==============================================================================
# Author : Kent Chiu (kentchun33333@gmail.com)
#==============================================================================

import cv2
from com_func.video_io import video_saving_IO, video_saving_CV
import logging
import argparse


#=======================================================================
# Set up a logger thead name, could be used in multi-threading condition

parser = argparse.ArgumentParser()
# Add the video path
parser.add_argument('-o','--output', required=True,
        help='output the vid_path Name & logging Name')
# Add the init bounding box position
parser.add_argument('-i','--initBox',
        help='designation the location of the tracking object')
args = vars(parser.parse_args())



camera = cv2.VideoCapture(0)

imageSeq = []
# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    cv2.imshow("Cou", frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ##cv2.imshow("CheckHSV", img_check)
    imageSeq.append(frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

video_saving_IO(fileName+'.avi', imageSeq)


# python webcam_record.py -o test_1006_03

