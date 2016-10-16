

#==============================================================================
# Author : Kent Chiu (kentchun33333@gmail.com)
#==============================================================================
#
# Notice
# - This recipe is used to test dlib state-of-art tracking algorithm
#
# ================================================================
# Usage Example
# ================================================================
# python check_initPosition.py -i test_1006.avi -b 0 270 150 450
# python check_initPosition.py -i test_1006_02.avi -b 230 220 430 450
# python check_initPosition.py -i test_1006_03.avi -b 260 230 445 410
# python check_initPosition.py -i test_1006_04.avi -b 110 170 345 405
#==============================================================================


import dlib
import argparse
import imageio
import time
import gc
import cv2

def to_int(*args):
    result=[]
    for i in args:
        result.append(int(i))
    return result

# ======================================================
# Define

parser = argparse.ArgumentParser()
# Add the video path
parser.add_argument('-i', '--input', required=True)

# Add the init bounding box position
parser.add_argument('-b','--tarBox',required=True, type=int, nargs='+')
args = vars(parser.parse_args())


vid = imageio.get_reader(args['input'])

# Get the initial frame as the reference
img = vid.get_data(0)

# Init the tracker class
tracker = dlib.correlation_tracker()

a,b,c,d = args['tarBox']
# A little larger init bounding box would be better : )
tracker.start_track(img, dlib.rectangle(a,b,c,d))


win = dlib.image_window()
for i in range(1,30):
    # Failure Reproduce Success
    # Reinit the tracker would cause the threading problem

    img = vid.get_data(i)
    tracker.update(img)
    #tracker.get_position()
    d = tracker.get_position()
    startX, startY, endX, endY = to_int(d.left(), d.top(),d.right(),d.bottom())
    print ('{},{},{},{}'.format(startX, startY, endX, endY) )
    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(tracker.get_position())
    gc.collect()
    dlib.hit_enter_to_continue()
    time.sleep(0.005)




