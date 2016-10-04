
#==============================================================================
# Author : Kent Chiu (kentchun33333@gmail.com)
#==============================================================================
#
# Notice
# - This recipe is used to test dlib state-of-art tracking algorithm
# - And embedded into python Class for use
#
#==============================================================================



import dlib
import unittest
import time
import imageio

class Tracker(object):
    def __init__(self):
        self.tracker=dlib.correlation_tracker()
    
    def start_track(self, *arg, **kwargs):
        return self.tracker.start_track(*arg, **kwargs)
    
    def update(self,*arg, **kwargs):
        return self.tracker.update(*arg, **kwargs)

    def get_position(self,*arg, **kwargs):
        return self.tracker.get_position(*arg, **kwargs)


class Test_Case(unittest.TestCase):
    def test_teacker(self):
        tracker = Tracker()
        self.assertEqual(hasattr(tracker, 'start_track'),True)


def to_int(*args):
    result=[]
    for i in args:
        result.append(int(i))
    return result


if __name__=='__main__':
    vid = imageio.get_reader('/Users/kentchiu/MIT_Vedio/'
        '2016-01-21/10.167.10.158_01_20160121082638418_1.mp4')
    # Get the initial frame as the reference
    img = vid.get_data(0)
    # Init the tracker class
    tracker = Tracker()

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
