__Author__ ='Kent Chiu'

# This code combined the feature of 
# Hand Contour Detection, with the ability to show different gesture
# implement by SKin_Hand_Detection class
# - MIT2016/Motion_Recognition/Hand_detection_0706.py
# also adapted to the Mot2Reg class for Multi_object_Reg2Mot

import cv2
import numpy as np
import argparse
import motion_interpretation
import imageio
from scipy.stats.mstats import mode as statics_mode
import imutils
from HUB_dictionary.motion_dictionary  import Motion_Dictionary
from HUB_dictionary.path_dictionary    import Path_DTW_Dictionary
from HUB_dictionary.gesture_dictionary import Gesture_DTW_Dictionary
from HUB_Model.multi_recog import SKin_Hand_Detection
from model_Mot2Reg import Recog2Track
from com_func.conf import Conf

def video_saving_IO(fileName, imgSequence):
    assert(fileName.split('.')[-1]=='avi')
    writer = imageio.get_writer(fileName)
    for image in imgSequence:
        writer.append_data(image)
    # Release everything if job is finished
    writer.close()
    print ('[*] Finish Saving {} at {}'.format(fileName, os.pardir.join([os.getcwd(),fileName])))

def get_args():
    parser = argparse.ArgumentParser(
        description=''' Script read both depth and rgb frames from kinetic 
        and save them in Assigned Folder''')

    parser.add_argument('-fn', '--output_Video', type=str, help='xxxxx.avi')
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    output_Video = args.output_Video or None
    # Return all variable values
    return output_Video

#Open Camera object
cap = cv2.VideoCapture(0)

model_1 = SKin_Hand_Detection()
main_Recog2Track = Recog2Track([model_1],['Hand'], True)


output_Video = get_args()

if output_Video is not None:
    imgSequence = []

while(1):
    #Capture frames from the camera
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    result = main_Recog2Track.perform_WebCam_analysis(frame)
    if output_Video is not None:
        imgSequence.append(result)

    cv2.imshow("Motion Recognition : Press ESC to EXIT", result)

    #close the output video by pressing 'ESC'
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        if output_Video is not None:
            video_saving_IO(fileName, imgSequence)


cap.release()
cv2.destroyAllWindows()
