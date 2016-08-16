import cv2
import numpy as np 

import imageio

vid = imageio.get_reader('D:/2016-01-21//10.167.10.158_01_20160121082638418_1.mp4')

class ObjectTracking:
	def __init__(self, rawImg, templateImg):
		self.rawImg = rawImg
		self.tmpImg = templateImg
	def match_search(self):
		pass

'''
Lucas-Kanade method computes optical flow for a sparse feature set 
(in our example, corners detected using Shi-Tomasi algorithm). 

OpenCV provides another algorithm to find the dense optical flow. 
It computes the optical flow for all the points in the frame. 

It is based on Gunner Farnebacks algorithm which is explained in 
"Two-Frame Motion Estimation Based on Polynomial Expansion" 
by Gunner Farneback in 2003.
Below sample shows how to find the dense optical flow using above algorithm. 
We get a 2-channel array with optical flow vectors, (u,v). 

We find their magnitude and direction. 
We color code the result for better visualization. 

Direction corresponds to Hue value of the image.
Magnitude corresponds to Value plane.
'''

import cv2
import numpy as np

#cap = cv2.VideoCapture("vtest.avi")
#ret, frame1 = cap.read()
frame1 = vid.get_data(0)
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

maxFrame = 500
frameID=1
while(frameID<maxFrame):
    #ret, frame2 = cap.read()
    frame2 = vid.get_data(frameID)

    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    if int(cv2.__version__[0])==3:
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    else:
        flow = cv2.calcOpticalFlowFarneback(prvs,next, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2',bgr)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',bgr)
    prvs = next
    frameID+=1

#cap.release()
cv2.destroyAllWindows()

'''
LK method
'''

def lk_shiTomasi(vid, startFrame, endFrame, frameBuff):
    '''startFrame int , endFrame int, frameBuff int
    '''
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    #ret, old_frame = cap.read()
    old_frame = vid.get_data(startFrame)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    frameID=startFrame+1
    while(frameID<endFrame):
        frame = vid.get_data(frameID)
        frameID+=1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params) 

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]    

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)   

        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break   

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    cv2.destroyAllWindows()
#cap.release()


















# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
#ret, old_frame = cap.read()
old_frame = vid.get_data(0)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

maxFrame = 500
frameID=1
while(frameID<maxFrame):
    frame = vid.get_data(frameID)
    frameID+=1
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        #mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        #frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imshow('frame',mask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
#cap.release()