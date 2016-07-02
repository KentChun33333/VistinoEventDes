'''
@ Author : JIN-CHUN CHIU (Kent)
This module contains several tool for interacting tuning Haar Feature on Hand.
'''

import cv2
import imageio
import os
from skimage import io 

##############################################################################
# Variables Configuation
##############################################################################
vid= imageio.get_reader('10.167.10.158_01_20160121082638418_1.mp4')
img = vid.get_data(1)

############################################################################## 
# Functions
##############################################################################
def testing_result(obj):
    '''obj = hand2_cascade.detectMultiScale(closed) '''
    a = img.copy
    for (x,y,w,h) in obj:
        # draw rectangle at the specific location
        cv2.rectangle(a,(x,y),(x+w,y+h),(255,0,0),2) 
        # extract the segmentation 
    io.imshow(a)

def testing_result_2(obj, obj2):
    '''obj = hand2_cascade.detectMultiScale(closed) '''
    a = obj2
    for (x,y,w,h) in obj:
        # draw rectangle at the specific location
        cv2.rectangle(a,(x,y),(x+w,y+h),(255,0,0),2) 
        # extract the segmentation 
    io.imshow(a)

ref = img.copy()
gray = cv2.cvtColor(img.copy()[100:500,50:800], cv2.COLOR_BGR2GRAY)


hand1_cascade = cv2.CascadeClassifier(
    '/Users/kentchiu/Opencv-1/haarcascade/aGest.xml')
hand2_cascade = cv2.CascadeClassifier(
    '/Users/kentchiu/Opencv-1/haarcascade/palm.xml')
hand3_cascade = cv2.CascadeClassifier(
    '/Users/kentchiu/Opencv-1/haarcascade/Hand.Cascade.1.xml')
hand4_cascade = cv2.CascadeClassifier(
    '/Users/kentchiu/Dropbox/Screenshots/Haar_hand/result/Cascade.xml')

hand_5 = cv2.CascadeClassifier(
    '/Users/kentchiu/MIT_Vedio/Rhand_no_tools/cascade.xml')

# Tuning 

hand4 = hand4_cascade.detectMultiScale(
    img.copy()[100:500,50:800],
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(60, 80),
    maxSize=(100, 180),
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

hand4 = hand4_cascade.detectMultiScale(
    img.copy()[100:500,50:800],
    scaleFactor=1.2,
    minNeighbors=515,
    minSize=(50, 80),
    maxSize=(100, 180),
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )


hand4 = hand4_cascade.detectMultiScale(
    img.copy()[100:500,50:800],
    scaleFactor=1.2,
    minNeighbors=515,
    minSize=(50, 80),
    maxSize=(100, 180),
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )


for (x,y,w,h) in hand4:
    # draw rectangle at the specific location
    cv2.rectangle(ref,(x+50,y+100),(x+50+w,y+100+h),(255,0,0),2) 
    # extract the segmentation 
    roi_gray = gray[y:y+h, x:x+w]



# MAy 2 Test
# Test 1 => not work 
hand5 = hand_5.detectMultiScale(
    img.copy(),
    scaleFactor=1.2,
    minNeighbors=100,
    minSize=(50, 80),
    maxSize=(100, 180),
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )


for (x,y,w,h) in hand5:
    # draw rectangle at the specific location
    cv2.rectangle(ref,(x,y),(x+w,y+h),(255,0,0),2) 
    # extract the segmentation 
    roi_gray = gray[y:y+h, x:x+w]


# Test 2
hand5 = hand_5.detectMultiScale(
    img.copy()
    scaleFactor=1.2,
    minNeighbors=35, #35 ==> 1
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

for (x,y,w,h) in hand5:
    # draw rectangle at the specific location
    cv2.rectangle(ref,(x,y),(x+w,y+h),(255,0,0),2) 




# Test 3
img = vid.get_data(1)
ref = img.copy

hand5 = hand_5.detectMultiScale(
    img.copy()[100:500,50:800],
    scaleFactor=1.2,
    minNeighbors=25, #35 ==> 1
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )


for (x,y,w,h) in hand5:
    # draw rectangle at the specific location
    cv2.rectangle(ref,(x+50,y+100),(x+50+w,y+100+h),(255,0,0),2) 



# Test 4

img4 = vid.get_data(65)
ref  = img4.copy()

hand5 = hand_5.detectMultiScale(
    img4.copy()[50:550,50:800],
    scaleFactor=1.2,
    minNeighbors=25, #35 ==> 1
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
if len(hand5)>1:
	minNeighbors+=5
if len(hand5)==1:
	# do something : )


for (x,y,w,h) in hand5:
    # draw rectangle at the specific location
    cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 

io.imshow(ref)



# Test 5 : work
img6 = vid.get_data(100)
ref  = img5.copy()

hand5 = hand_5.detectMultiScale(
    ref.copy()[50:550,50:800],
    scaleFactor=1.2,
    minNeighbors=25, #35 ==> 1
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
for (x,y,w,h) in hand5:
    # draw rectangle at the specific location
    cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 

io.imshow(ref)

# Test 6 : work
img6 = vid.get_data(150)


ref  = img6.copy()

hand5 = hand_5.detectMultiScale(
    ref.copy()[50:550,50:800],
    scaleFactor=1.2,
    minNeighbors=25, #35 ==> 1
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
for (x,y,w,h) in hand5:
    # draw rectangle at the specific location
    cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 
io.imshow(ref)



# Test 7 : 

def cascade_test(img):
    ref = img.copy()
    hand5 = hand_5.detectMultiScale(
        ref.copy()[50:550,50:800],
        scaleFactor=1.2,
        minNeighbors=25, #35 ==> 1
        #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    for (x,y,w,h) in hand5:
        # draw rectangle at the specific location
        cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 
    io.imshow(ref)

############################
# Key fail at 330 328 327  #
############################

# Testing 8
img = vid.get_data(330)


ref  = img.copy()

hand5 = hand_5.detectMultiScale(
    ref.copy()[50:550,50:800],
    scaleFactor=1.2,
    minNeighbors=15, #35 ==> 1
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
for (x,y,w,h) in hand5:
    # draw rectangle at the specific location
    cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 
io.imshow(ref)

# OK

# Testing 9
def cascade_test_15(img):
    ref = img.copy()
    hand5 = hand_5.detectMultiScale(
        ref.copy()[50:550,50:800],
        scaleFactor=1.2,
        minNeighbors=15, #35 ==> 1
        #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    for (x,y,w,h) in hand5:
        # draw rectangle at the specific location
        cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 
    io.imshow(ref)

# Testing 10
def cascade_test_20(img):
    ref = img.copy()
    hand5 = hand_5.detectMultiScale(
        ref.copy()[50:550,50:800],
        scaleFactor=1.2,
        minNeighbors=20, #35 ==> 1
        #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    for (x,y,w,h) in hand5:
        # draw rectangle at the specific location
        cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 
    io.imshow(ref)


# Testing 11
def cascade_test_25(img):
    ref = img.copy()
    hand5 = hand_5.detectMultiScale(
        ref.copy()[50:550,50:800],
        scaleFactor=1.2,
        minNeighbors=25, #35 ==> 1
        #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    for (x,y,w,h) in hand5:
        # draw rectangle at the specific location
        cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 
    io.imshow(ref)


# Testing 12
def cascade_test_17(img):
    ref = img.copy()
    hand5 = hand_5.detectMultiScale(
        ref.copy()[50:550,50:800],
        scaleFactor=1.2,
        minNeighbors=17, #35 ==> 1
        #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    hand_position=[]
    for (x,y,w,h) in hand5:
        # draw rectangle at the specific location
        cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 
        # Recording the hand_position
        hand_position.append((((x+100+x+w)/2),((y+100+y+h)/2)))
    io.imshow(ref)
    print hand_position


# Base-on Testing 12 -> out put the hand_postion
def cascade_test_17_return(img):
    ref = img.copy()
    hand5 = hand_5.detectMultiScale(
        ref.copy()[50:550,50:800],
        scaleFactor=1.2,
        minNeighbors=17, #35 ==> 1
        #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    hand_position=[]
    for (x,y,w,h) in hand5:
        # draw rectangle at the specific location
        cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 
        # Recording the hand_position
        hand_position.append((((x+100+x+w)/2),((y+100+y+h)/2)))
    #io.imshow(ref)
    return hand_position




# Testing Script 
hand_position_sum=[]
for i in range(500):
    img = vid.get_data(i)
    hand_position_sum+=[cascade_test_17_return(img)]

# recode
file = open('test.txt', 'w')

frame_idx=0
vid_name = '10.167.10.158_01_20160121082638418_1.mp4'
mode_name = 'T12-bea'
file.write('vid_name,'+'mode_name,'+'frame_idx,'+'hand_position'+'\n')
for i in hand_position_sum:
    i = str(i).replace(',','-')
    file.write(vid_name+','+mode_name+','+str(frame_idx)+','+str(i)+'\n')
    frame_idx+=1
file.close()









