'''
@ Author : Chiu, JIN-CHUN (Kent)
MP4 Data Extraction +
Object Tracking with Hands with White Gloves
'''

111111111111111111111111111111111111111111111111111111111111111111111122
########################################################################
import cv2
import imageio
import skimage
import os
from skimage import io 
'''
Script Zone
'''



H_hsv = s[0,0:,0]
S_hsv = s[0,0:,1]
V_hsv = s[0,0:,2]


import cv2

cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

'''
x1,y1 ------
|          |
|          |
|          |
--------x2,y2
'''


def HSV_range_probability(temp_list):
    temp = {}
    for i in set(temp_list):
        temp[i]=1
    for i in temp_list:
        temp[i]+=1
    return temp
h1 = HSV_range_probability(H_hsv)
s1 = HSV_range_probability(S_hsv)
v1 = HSV_range_probability(V_hsv)




def read_mp4(filepath):
	vid = imageio.get_reader(filepath)
	frame = vid.get_data(1)
	io.imshow(frame)
	ret, thresh1 = cv2.threshold(frame, 255, 255, cv2.THRESH_BINARY)


for i in [155, 200, 250]:
	ret, thresh1 = cv2.threshold(frame, 250, i, cv2.THRESH_BINARY)
	io.imshow(thresh1)

def white_extraction(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of white color in HSV
    # change it according to your need !
    lower_white = np.array([0,0,0], dtype=np.uint8)
    upper_white = np.array([0,0,255], dtype=np.uint8)
    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    return res

def background_substraction(filepath):
    #fgbg = cv2.createBackgroundSubtractorMOG()
    fgbg = cv2.createBackgroundSubtractorMOG2()
    # training with frames
    vid = imageio.get_reader(filepath)
    for i in range(100):
        fgmask = fgbg.apply(vid.get_data(i))
    cv2.imshow('frame',fgmask)
    return 
'''
import cv2
import numpy as np

c = cv2.VideoCapture('test.avi')
_,f = c.read()

avg1 = np.float32(f)
avg2 = np.float32(f)

# loop over images and estimate background 
for x in range(0,4):
    _,f = c.read()

    cv2.accumulateWeighted(f,avg1,1)
    cv2.accumulateWeighted(f,avg2,0.01)

    res1 = cv2.convertScaleAbs(avg1)
    res2 = cv2.convertScaleAbs(avg2)

    cv2.imshow('img',f)
    cv2.imshow('avg1',res1)
    cv2.imshow('avg2',res2)
    k = cv2.waitKey(0) & 0xff
    if k == 5:
        break

BackgroundSubtractorMOG
It is a Gaussian Mixture-based Background/Foreground Segmentation Algorithm. It was introduced in the paper “An improved adaptive background mixture model for real-time tracking with shadow detection” by P. KadewTraKuPong and R. Bowden in 2001. It uses a method to model each background pixel by a mixture of K Gaussian distributions (K = 3 to 5). The weights of the mixture represent the time proportions that those colours stay in the scene. The probable background colours are the ones which stay longer and more static.

While coding, we need to create a background object using the function, cv2.createBackgroundSubtractorMOG(). It has some optional parameters like length of history, number of gaussian mixtures, threshold etc. It is all set to some default values. Then inside the video loop, use backgroundsubtractor.apply() method to get the foreground mask.

See a simple example below:

import numpy as np
import cv2

cap = cv2.VideoCapture('vtest.avi')

fgbg = cv2.createBackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
( All the results are shown at the end for comparison).

BackgroundSubtractorMOG2
It is also a Gaussian Mixture-based Background/Foreground Segmentation Algorithm. It is based on two papers by Z.Zivkovic, “Improved adaptive Gausian mixture model for background subtraction” in 2004 and “Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction” in 2006. One important feature of this algorithm is that it selects the appropriate number of gaussian distribution for each pixel. (Remember, in last case, we took a K gaussian distributions throughout the algorithm). It provides better adaptibility to varying scenes due illumination changes etc.

As in previous case, we have to create a background subtractor object. Here, you have an option of selecting whether shadow to be detected or not. If detectShadows = True (which is so by default), it detects and marks shadows, but decreases the speed. Shadows will be marked in gray color.

import numpy as np
import cv2

cap = cv2.VideoCapture('vtest.avi')

fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
(Results given at the end)

BackgroundSubtractorGMG
This algorithm combines statistical background image estimation and per-pixel Bayesian segmentation. It was introduced by Andrew B. Godbehere, Akihiro Matsukawa, Ken Goldberg in their paper “Visual Tracking of Human Visitors under Variable-Lighting Conditions for a Responsive Audio Art Installation” in 2012. As per the paper, the system ran a successful interactive audio art installation called “Are We There Yet?” from March 31 - July 31 2011 at the Contemporary Jewish Museum in San Francisco, California.

It uses first few (120 by default) frames for background modelling. It employs probabilistic foreground segmentation algorithm that identifies possible foreground objects using Bayesian inference. The estimates are adaptive; newer observations are more heavily weighted than old observations to accommodate variable illumination. Several morphological filtering operations like closing and opening are done to remove unwanted noise. You will get a black window during first few frames.

It would be better to apply morphological opening to the result to remove the noises.

import numpy as np
import cv2

cap = cv2.VideoCapture('vtest.avi')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorGMG()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
'''

def draw_window(frame):
    # setup initial location of window
    r,h,c,w = 250,90,400,125  # simply hardcoded the values
    track_window = (c,r,w,h)    

    # set up the ROI for tracking
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)    

    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

    # apply meanshift to get the new location
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)
    # Draw it on image
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    img2 = cv2.polylines(frame,[pts],True, 255,2)
    io.imshow(img2)



class Vid():
	def __init__(self, filepath):
		self.filepath=filepath
	def getframes(self, filepath):
		vid = imageio.get_reader(filepath)


'''
camshift_tracking

'''
import cv2
import cv2.cv as cv

bb = (100,200,100,100)

def camshift_tracking(img1, img2, bb):
        hsv = cv2.cvtColor(img1, cv.CV_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        x0, y0, w, h = bb
        x1 = x0 + w -1
        y1 = y0 + h -1
        hsv_roi = hsv[y0:y1, x0:x1]
        mask_roi = mask[y0:y1, x0:x1]
        hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX);
        hist_flat = hist.reshape(-1)
        prob = cv2.calcBackProject([hsv,cv2.cvtColor(img2, cv.CV_BGR2HSV)], [0], hist_flat, [0, 180], 1)
        prob &= mask
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        new_ellipse, track_window = cv2.CamShift(prob, bb, term_crit)
        print track_window
        print new_ellipse
        return track_window
 
def face_track():
    cap = cv2.VideoCapture(0)
    img = cap.read()
    bb = (125,125,200,100) # get bounding box from some method

    while True:
        try:
            img1 = cap.read()
            bb = camshift(img1, img, bb)
            img = img1
            #draw bounding box on img1
            imshow("CAMShift",img1)
        except KeyboardInterrupt:
            break





'''
I wrote this for tracking white color :
'''
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    # change it according to your need !
    lower_white = np.array([0,0,0], dtype=np.uint8)
    upper_white = np.array([0,0,255], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

'''
END
'''



_, frame = cap.read()
# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# define range of blue color in HSV
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])
# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)
# Bitwise-AND mask and original image
res = cv2.bitwise_and(frame,frame, mask= mask)
cv2.imshow('frame',frame)
cv2.imshow('mask',mask)
cv2.imshow('res',res)








vid = imageio.
'''

'''


import numpy as np
import cv2

cap = cv2.VideoCapture('slow.flv')

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
r,h,c,w = 250,90,400,125  # simply hardcoded the values
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)

    else:
        break

cv2.destroyAllWindows()
cap.release()





'''
Camshift in OpenCV

It is almost same as meanshift, but it returns a rotated rectangle (that is our result)
 and box parameters (used to be passed as search window in next iteration). See the code below:
'''
import numpy as np
import cv2

cap = cv2.VideoCapture('slow.flv')

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
r,h,c,w = 250,90,400,125  # simply hardcoded the values
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True, 255,2)
        cv2.imshow('img2',img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)

    else:
        break

cv2.destroyAllWindows()
cap.release()





        for j in range(len(res[0][0])):
            if res[i][j]>1:
                print i, j
                
white_extraction(frame)
res
track_window = ( c,r,w,h)_
track_window = ( c,r,w,h)
track_window = ( c=250,r=90,w=400,h=125)
track_window = ( 250,90,400,125)
r = 250
r,h,c,w = 250,90,400,125  # simply hardcoded the values
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
erm_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
ret, track_window = cv2.CamShift(dst, track_window, term_crit)
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
ret, track_window = cv2.CamShift(dst, track_window, term_crit)
ret
io.imshow(ret)
pts = cv2.boxPoints(ret)
pts = np.int0(pts)
img2 = cv2.polylines(frame,[pts],True, 255,2)
cv2.imshow('img2',img2)

def white_extraction(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of white color in HSV
    # change it according to your need !
    lower_white = np.array([0,0,0], dtype=np.uint8)
    upper_white = np.array([0,0,255], dtype=np.uint8)
    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    return res


def draw_window(frame):
    # setup initial location of window
    r,h,c,w = 250,90,400,125  # simply hardcoded the values
    track_window = (c,r,w,h)    

    # set up the ROI for tracking
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)    

    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

    # apply meanshift to get the new location
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)
    # Draw it on image
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    img2 = cv2.polylines(frame,[pts],True, 255,2)
    io.imshow(img2)
    
io.imshow(frame)
aaa = white_extraction(frame)
draw_window(aaa)
draw_window(frame)
draw_window(vid.get_data(10))
draw_window(vid.get_data(100))
draw_window(vid.get_data(1000))
draw_window(vid.get_data(120))
draw_window(vid.get_data(100))
draw_window(vid.get_data(1))
draw_window(vid.get_data(3))
draw_window(vid.get_data(4))
draw_window(vid.get_data(10))
draw_window(vid.get_data(30))
draw_window(vid.get_data(40))
draw_window(vid.get_data(4000))
draw_window(vid.get_data(4004))
draw_window(vid.get_data(14004))
draw_window(vid.get_data(12004))
draw_window(vid.get_data(12304))
draw_window(vid.get_data(12334))
draw_window(vid.get_data(11334))
draw_window(vid.get_data(12334))
draw_window(vid.get_data(22334))
draw_window(vid.get_data(12334))
import argparse
def camshift_tracking(img1, img2, bb):
        hsv = cv2.cvtColor(img1, cv.CV_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        x0, y0, w, h = bb
        x1 = x0 + w -1
        y1 = y0 + h -1
        hsv_roi = hsv[y0:y1, x0:x1]
        mask_roi = mask[y0:y1, x0:x1]
        hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX);
        hist_flat = hist.reshape(-1)
        prob = cv2.calcBackProject([hsv,cv2.cvtColor(img2, cv.CV_BGR2HSV)], [0], hist_flat, [0, 180], 1)
        prob &= mask
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        new_ellipse, track_window = cv2.CamShift(prob, bb, term_crit)
        return track_window

io.imshow(data)
io.imshow(es)
io.imshow(res)
data2=vid.get_data(100)
io.imshow(data2)
ls

##---(Mon Mar 21 08:41:41 2016)---
import cv2
import imageio
import skimage
import os
from skimage import io

cd MIT_Vedio/
cd 2016-01-21/
ls
def read_mp4(filepath):
    vid = imageio.get_reader(filepath)
    frame = vid.get_data(1)
    io.imshow(frame)
    ret, thresh1 = cv2.threshold(frame, 255, 255, cv2.THRESH_BINARY)
 
read_mp4('10.167.10.158_01_20160121082638418_1.mp4')
vid
vid=imageio.get_reader('10.167.10.158_01_20160121082638418_1.mp4')
vid.get_length
vid.get_length()
for i in [1,50,100,150,200,250]:
    print i
    
import cv2
iport numpy as np
import numpy as np
avg1=np.float32(f)
for i in [1,50,100,150,200,250]:
    code='''avg%s=vid.get_data(%s)'''%(str(i), i)
    
exec code
for i in [1,50,100,150,200,250]:
    code='''avg%s=vid.get_data(%s)'''%(str(i), i)
    exec code
    
from skimage import io
io.imshow(avg1-avg100)
io.imshow(avg1+avg100)
io.imshow(avg1+avg100+avg150)
io.imshow(avg1+avg100+avg150+avg200)
cv2.accumulateWeighted(vid.get_data(10), avg1, 1)
cv2.accumulateWeighted(vid.get_data(102), avg1, 1)
io.imshow(avg1-avg100)
io.imshow(reversed(avg1-avg100))
io.imshow((avg1+avg100))
fgbg= cv2.createBackgroundSubtractorKNN()
fgmask = fgbg.apply(avg1)
io.imshow(fgmask)
fgbg= cv2.createBackgroundSubtractorMOG2()
fgmask = fgbg.apply(avg1)
io.imshow(fgmask)
fgbg= cv2.createBackgroundSubtractorMOG2(1000)
fgbg.getBackgroundImage()
io.imshow(fgbg.getBackgroundImage())
io.imshow(fgmask)
fgbg= cv2.createBackgroundSubtractorMOG()
fgbg= cv2.createBackgroundSubtractorMOG2()
fgbg.apply(ivg1, ivg100)
fgbg.apply(avg1, avg100)
a = fgbg.apply(avg1, avg100)
io.imshow(a)
a = fgbg.apply(avg1)
io.imshow(a)
a = fgbg.apply(avg1)
a = fgbg.apply(avg100)
fgbg.clear()
fgbg.apply(avg100)
io.imshow(fgbg.apply(avg100))
cv2
cv2.version
cv2.version()
vid.get_data(1)
io.imshow(vid.get_data(1))
io.imshow(vid.get_data(2))
io.imshow(vid.get_data(20))
io.imshow(vid.get_data(30))
io.imshow(vid.get_data(130))
io.imshow(vid.get_data(230))
io.imshow(vid.get_data(330))
io.imshow(vid.get_data(230))
io.imshow(vid.get_data(240))
io.imshow(vid.get_data(200))
io.imshow(vid.get_data(180))
io.imshow(vid.get_data(250))
io.imshow(vid.get_data(300))
io.imshow(vid.get_data(310))
io.imshow(vid.get_data(320))
io.imshow(vid.get_data(310))
background = vid.get_data(310)
io.imshow(avg1-backround)
io.imshow(avg1-background)
fgbg= cv2.createBackgroundSubtractorMOG2()
fgbg.apply(background)
a = fgbg.apply(background)
imshow(a)
io.imshow(a)
io.imshow(avg1)
fgbg.getBackgroundImage()
io.imshow(fgbg.getBackgroundImage())
fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg.getBackgroundImage()
io.imshow(fgbg.getBackgroundImage())
fgbg.apply(background)
a = fgbg.apply(background)
a = fgbg.apply(avg1)
io.imshow(a)
a = fgbg.apply(avg100)
io.imshow(a)
a = fgbg.apply(avg150)
io.imshow(a)
for i in range(100):
    fgbg.apply(vid.get_data(i))
    
io.imshow(fgbg.apply(avg100))
io.imshow(fgbg.apply(avg150))
io.imshow(fgbg.apply(avg200))
io.imshow(fgbg.apply(avg250))
io.imshow(fgbg.apply(avg1))
io.imshow(fgbg.apply(avg50))

    io.imshow(fgbg.apply(vid.get_data(1000)))
io.imshow(fgbg.apply(vid.get_data(1000)))
io.imshow(fgbg.apply(vid.get_data(300)))
io.imshow(fgbg.apply(vid.get_data(400)))
io.imshow(fgbg.apply(vid.get_data(500)))
io.imshow(fgbg.apply(vid.get_data(600)))
io.imshow(fgbg.apply(vid.get_data(700)))
io.imshow(fgbg.apply(vid.get_data(800)))
io.imshow(fgbg.apply(vid.get_data(900)))
io.imshow(fgbg.apply(vid.get_data(1000)))
io.imshow(fgbg.apply(vid.get_data(1001)))
io.imshow(fgbg.apply(vid.get_data(1002)))
io.imshow(fgbg.apply(vid.get_data(1003)))
for i in range(100, 200):
    fgbg.apply(vid.get_data(i))
    
fgbg.apply(vid.get_data(201))
io.imshow(fgbg.apply(vid.get_data(201)))
io.imshow(fgbg.apply(vid.get_data(202)))
io.imshow(fgbg.apply(vid.get_data(203)))
io.imshow(fgbg.apply(vid.get_data(204)))
io.imshow(fgbg.apply(vid.get_data(2045))
)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg2 = cv2.createBackgroundSubtractorKNN()
for i in range(100, 200):
    fgbg2.apply(vid.get_data(i))
    
io.imshow(fgbg2.apply(vid.get_data(204)))
io.imshow(fgbg2.apply(vid.get_data(200)))
io.imshow(fgbg2.apply(vid.get_data(250)))
io.imshow(fgbg2.apply(vid.get_data(150)))
io.imshow(fgbg2.apply(vid.get_data(100)))
io.imshow(fgbg2.apply(vid.get_data(300)))
io.imshow(fgbg2.apply(vid.get_data(400)))
io.imshow(fgbg2.apply(vid.get_data(410)))
io.imshow(fgbg2.apply(vid.get_data(412)))
io.imshow(fgbg2.apply(vid.get_data(100)))
io.imshow(fgbg2.apply(vid.get_data(150)))
for i in range(200,300):
    fgbg2.apply(vid.get_data(i))
    
io.imshow(fgbg2.apply(vid.get_data(200)))
io.imshow(fgbg2.apply(vid.get_data(300)))
io.imshow(fgbg2.apply(vid.get_data(350)))
fgbg2.clear()
for i in range(250,300):
    fgbg2.apply(vid.get_data(i))
    
io.imshow(fgbg2.apply(vid.get_data(350)))
hsv = cv2.cvtColor(avg1, cv2.COLOR_BGR2HSV)
io.imshow(hsv)
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

mask = cv2.inRange(hsv, lower_blue, upper_blue)
io.imshow(mask)
lower_blue = np.array([110,0,50])
upper_blue = np.array([130,55,255])

mask = cv2.inRange(hsv, lower_blue, upper_blue)
io.imshow(mask)
io.imshow(hsv)
hsv[300,600]
lower_blue = np.array([0,0,0])
upper_blue = np.array([50,55,50])

mask = cv2.inRange(hsv, lower_blue, upper_blue)
io.imshow(mask)
mask = cv2.inRange(hsv, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
io.imshow(mask)
lower_white = np.array([0,0,0], dtype=np.uint8)
upper_white = np.array([0,0,255], dtype=np.uint8)

mask = cv2.inRange(hsv, lower_white, upper_white)
io.imshow(mask)
lower_white = np.array([0,0,0], dtype=np.uint8)
upper_white = np.array([10,10,255], dtype=np.uint8)

mask = cv2.inRange(hsv, lower_white, upper_white)
io.imshow(mask)
lower_white = np.array([0,0,0], dtype=np.uint8)
upper_white = np.array([20,20,255], dtype=np.uint8)

mask = cv2.inRange(hsv, lower_white, upper_white)
io.imshow(mask)
io.imshow(hsv[100:])
io.imshow(hsv[100:200])
io.imshow(hsv[300:500])
io.imshow(hsv[200:500])
io.imshow(hsv[200:500][400:800])
io.imshow(hsv[200:500],[400:800])
io.imshow(hsv[200:500,400:800])
io.imshow(hsv[250:450,400:800])
io.imshow(hsv[250:450,550:610])
io.imshow(hsv[250:450,550:710])
io.imshow(hsv[250:450,550:680])
io.imshow(hsv[250:450,580:680])
io.imshow(hsv[250:450,585:680])
io.imshow(hsv[240:450,585:680])
io.imshow(hsv[280:450,585:680])
io.imshow(hsv[290:450,585:680])
io.imshow(hsv[300:450,585:680])
io.imshow(hsv[310:450,585:680])
s = hsv[310:450,585:680]
s
min(s)
s[0]
s[0][0]
s[0][0:]
s[0:][0]
s[0,0]
s[0,0,0]
s[0,0,0:]
s[0,0:,0]
H_hsv = s[0,0:,0]
S_hsv = s[0,0:,1]
V_hsv = s[0,0:,2]
s
s = hsv[310:450,585:680]
io.imshow(s)
io.imshow(hsv[300:450,585:680])
io.imshow(hsv[320:450,585:680])
io.imshow(hsv[320:400,585:680])
io.imshow(hsv[320:410,585:680])
io.imshow(hsv[320:420,585:680])
io.imshow(hsv[320:420,590:680])
io.imshow(hsv[320:420,600:680])
io.imshow(hsv[320:420,600:670])
s = hsv[320:420,600:670]
H_hsv = s[0,0:,0]
S_hsv = s[0,0:,1]
V_hsv = s[0,0:,2]
min(H_hsv)
mzx(H_hsv)
max(H_hsv)
temp = {}
temp = {}
temp.keys=set(H_hsv)

for i in H_hsv:
    if temp[i]:
        temp[i]+=1
    else:
        temp[i]=1
        
for i in H_hsv:
    if temp[i] is not None:
        temp[i]+=1
    else:
        temp[i]=1
        
for i in H_hsv: print i
for i in H_hsv: 
    temp[i]=1
    
temp
for i in H_hsv:
    if type(temp[i]) is int:
        temp[i]+=1
    else:
        temp[i]=1
        
temp
temp = {}
for i in H_hsv:
    if type(temp[i]) is int:
        temp[i]+=1
    else:
        temp[i]=1
        
def HSV_range_probability(temp_list):
    temp = {}
    for i in set(temp_list):
        temp[i]=1
    for i in temp_list:
        temp[i]+=1
    print temp
HSV_range_probability(H_hsv)
HSV_range_probability(S_hsv)
HSV_range_probability(V_hsv)

def HSV_range_probability(temp_list):
    temp = {}
    for i in set(temp_list):
        temp[i]=1
    for i in temp_list:
        temp[i]+=1
    return temp
h1 = HSV_range_probability(H_hsv)
s1 = HSV_range_probability(S_hsv)
v1 = HSV_range_probability(V_hsv)

h1
s1
v1
lower_white = np.array([0,0,0], dtype=np.uint8)
upper_white = np.array([10,10,255], dtype=np.uint8)

mask = cv2.inRange(hsv, lower_white, upper_white)
io.imshow(mask)
mask[400:600,500:700]
mask2 = mask[320:420,600:670]
io.imshow(mask2)
lower_white = np.array([0,0,0], dtype=np.uint8)
upper_white = np.array([170,10,255], dtype=np.uint8)

io.imshow(mask2)
mask2 = mask[320:420,600:670]
io.imshow(mask2)
io.imshow(mask)
lower_white = np.array([10,10,0], dtype=np.uint8)
upper_white = np.array([170,170,255], dtype=np.uint8)

mask = cv2.inRange(hsv, lower_white, upper_white)
io.imshow(mask)
lower_white = np.array([10,10,0], dtype=np.uint8)
upper_white = np.array([100,100,255], dtype=np.uint8)

mask = cv2.inRange(hsv, lower_white, upper_white)
io.imshow(mask)
mask2 = mask[320:420,600:670]
io.imshow(mask2)
io.imshow(a)
fgbg.clear()
for i in range(100):
    fgmask = fgbg.apply(vid.get_data(i))
    
io.imshow(fgmask)
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)[-2]
io.imshow(cnts)
cnts
io.imshow(a)
io.imshow(fgmask)
fgmask=fgmask[200:500, ]
io.imshow(fgmask)
io.imshow(a)
io.imshow(b)
io.imshow(avg1)
b = avg1[310:450,585:680]
io.imshow(b)
b
hsv_b = cv2.cvtColor(b, cv2.COLOR_RGB2HSV)
io.imshow(hsv_b)
hsv_b2 = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
io.imshow(hsv_b2)
io.imshow(b)
H_hsv = b[0,0:,0]
S_hsv = b[0,0:,1]
V_hsv = b[0,0:,2]

h1 = HSV_range_probability(H_hsv)
s1 = HSV_range_probability(S_hsv)
v1 = HSV_range_probability(V_hsv)





GLOVE!!!!

lower_blue = np.array([0,0,0])
upper_blue = np.array([200,50,300])

lower_blue = np.array([0,0,100])
upper_blue = np.array([200,50,300])

lower_blue = np.array([0,0,200])
upper_blue = np.array([300,100,300])
mask = cv2.inRange(hsv, lower_white, upper_white)


gray = cv2.cvtColor(mask, cv2.COLOR_BAYER_GB2GRAY)
gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# blur and threshold the image
blurred=cv2.blur(gray,(9,9))

blurred = cv2.blur(gradient, (9, 9))
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)


image, contours,hierarchy= cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
img= cv2.drawContours(avg1, contours, -1,(0,0,0),3)


 '''
 '''









# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
cnts= cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = np.int0(cv2.cv.BoxPoints(rect))

# draw a bounding box arounded the detected barcode and display the
# image
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)






