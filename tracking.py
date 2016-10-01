

#==============================================================================
# Author : Kent Chiu (kentchun33333@gmail.com)
#==============================================================================
#
# Notice
# - This recipe is used to test dlib state-of-art tracking algorithm
#
#==============================================================================


import dlib
import argparse
import imageio
import time

class ObjectTrack(object):
    def __init__(self, ):
        pass

def get_cli():
    parser = argparse.ArgumentParser()
    # Add the video path
    parser.add_argument('-v','--vidPath',
        help='designation the vid_path')
    # Add the init bounding box position
    parser.add_argument('-i','--initBox',
        help='designation the location of the tracking object')
    args = vars(ap.parse_args())
    return args

def to_int(*args):
    result=[]
    for i in args:
        result.append(int(i))
    return result

vid = imageio.get_reader('/Users/kentchiu/MIT_Vedio/'
    '2016-01-21/10.167.10.158_01_20160121082638418_1.mp4')

# Get the initial frame as the reference
img = vid.get_data(0)

# Init the tracker class
tracker = dlib.correlation_tracker()

# A little larger init bounding box would be better : )
tracker.start_track(img, dlib.rectangle(580, 290, 700, 450))


win = dlib.image_window()
for i in range(1,500):
    img = vid.get_data(i)
    tracker.update(img)
    #tracker.get_position()
    d = tracker.get_position()
    startX, startY, endX, endY = to_int(d.left(), d.top(),d.right(),d.bottom())
    print ('{},{},{},{}'.format(startX, startY, endX, endY) )
    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(tracker.get_position())
    dlib.hit_enter_to_continue()
    time.sleep(0.005)




