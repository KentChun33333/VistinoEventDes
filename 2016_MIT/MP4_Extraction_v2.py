'''
##############################################################################
# @ Author : Chiu, JIN-CHUN (Kent)
# @ Goal :
# MP4 Data Extraction +
# Object Tracking with Hands with White Gloves
##############################################################################
'''


import cv2
import imageio
import os
from skimage import io 



'''
Step 1 : Hand Track with HSV
'''

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    # change it according to your need !
    lower_white = np.array([0,0,200])
    upper_white = np.array([300,100,300])

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
cap.release()






cap = cv2.VideoCapture(0)

while(1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    # change it according to your need !
    lower_white = np.array([0,0,200])
    upper_white = np.array([300,100,300])

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
'''
'''
hsv = cv2.cvtColor(avg1, cv2.COLOR_BGR2HSV)

GLOVE!!!!

lower_blue = np.array([0,0,0])
upper_blue = np.array([200,50,300])

lower_blue = np.array([0,0,100])
upper_blue = np.array([200,50,300])

lower_blue = np.array([0,0,200])
upper_blue = np.array([300,100,300])
mask = cv2.inRange(hsv, lower_white, upper_white)

##############################################################################
def test(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0,0,200])
    upper_white = np.array([300,100,300])
    # Threshold the HSV image to get only white colors
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

    image, contours,hierarchy= cv2.findContours(closed,
                               cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img= cv2.drawContours(frame, contours, -1,(0,0,0),3)
    io.imshow(img)

##############################################################################
def test2(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0,0,100])
    upper_white = np.array([200,50,300])
    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    gray = cv2.cvtColor(mask, cv2.COLOR_BAYER_GB2GRAY)
    gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    # blur and threshold the image
    blurred=cv2.blur(gray,(9,9))
    ###blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)

    image, contours,hierarchy= cv2.findContours(closed,
                               cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img= cv2.drawContours(frame, contours, -1,(0,0,0),3)
    io.imshow(img)
##############################################################################
def check_hsv1(frame):
    '''
    input : raw RGB frame
    output : RGB with contours ( white + blur + erode + dilate )
    '''
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0,0,100])
    upper_white = np.array([200,50,300])
    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    gray = cv2.cvtColor(mask, cv2.COLOR_BAYER_GB2GRAY)
    # blur and threshold the image
    blurred=cv2.blur(gray,(9,9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)
    #
    image, contours,hierarchy= cv2.findContours(closed,
                                                cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    img= cv2.drawContours(frame, contours, -1,(0,0,0),3)
    return img

##############################################################################
def check_hsv2(frame):
    '''
    in : raw RGB frame
    out : GRAY ( white + blur + erode + dilate )
    Note this GRAY img is useful to have Further Use like HAAR ...etc
    '''
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0,0,100])
    upper_white = np.array([200,50,300])
    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    gray = cv2.cvtColor(mask, cv2.COLOR_BAYER_GB2GRAY)
    # blur and threshold the image
    blurred=cv2.blur(gray,(9,9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)
    return closed


##############################################################################
fgbg= cv2.createBackgroundSubtractorMOG2()

fgbg.apply(background)

for i in range(30):
    fgbg.apply(vid.get_data(i))

frame_without_background = fgbg.apply(vid.get_data(35))

frame_without_background = d

blurred=cv2.blur(frame_without_background,(9,9))
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)



f = cv2.bitwise_and( check_hsv2(vid.get_data(35)), frame_without_background)
blurred=cv2.blur(f,(9,9))
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


def background_sub(frame): 
    '''
    out 
    '''
    return gray

    io.imshow(img)


'''
##############################################################################
Below code is use other's training HAAR xml file
'''

vid= imageio.get_reader('10.167.10.158_01_20160121082638418_1.mp4')
a= vid.get_data(100)

#
# Many operation in opencv is gray-scale
#
gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)


hand1_cascade = cv2.CascadeClassifier(
    '/Users/kentchiu/Opencv-1/haarcascade/aGest.xml')
hand2_cascade = cv2.CascadeClassifier(
    '/Users/kentchiu/Opencv-1/haarcascade/palm.xml')
hand3_cascade = cv2.CascadeClassifier(
    '/Users/kentchiu/Opencv-1/haarcascade/Hand.Cascade.1.xml')
hand4_cascade = cv2.CascadeClassifier(
    '/Users/kentchiu/Dropbox/Screenshots/Haar_hand/result/Cascade.xml')


# Combined with hsv selection
hand1 = hand1_cascade.detectMultiScale(check_hsv2(a)) 
hand2 = hand2_cascade.detectMultiScale(check_hsv2(a)) 
hand3 = hand3_cascade.detectMultiScale(check_hsv2(a)) 
hand4 = hand4_cascade.detectMultiScale(check_hsv2(a)) # not work...
hand4_1 = hand4_cascade.detectMultiScale(a) 


# Combined with hsv selection
hand1 = hand1_cascade.detectMultiScale(gray)
hand2 = hand2_cascade.detectMultiScale(gray) 
hand3 = hand3_cascade.detectMultiScale(gray) 
hand4 = hand4_cascade.detectMultiScale(gray) # not work...
hand4_1 = hand4_cascade.detectMultiScale(a) 



'''
##############################################################################
# 
##############################################################################

void CascadeClassifier::detectMultiScale(
    const Mat& image, 
    vector<Rect>& objects, 
    double scaleFactor=1.1,
    int minNeighbors=3, 
    int flags=0, 
    Size minSize=Size(),
    Size maxSize=Size() )


Amongst these parameters, you need to pay more attention to four of them:

[*] scaleFactor – 
    Parameter specifying how much the image size is reduced at each image scale.
    Basically the scale factor is used to create your scale pyramid. 
    More explanation can be found here. In short, as described here, 
    your model has a fixed size defined during training, which is visible in the xml. 
    
    This means that this size of face is detected in the image if present. 
    However, by rescaling the input image, you can resize a larger face to a smaller one, 
    making it detectable by the algorithm.
    
    1.05 is a good possible value for this, which means you use a small step for resizing, i.e. 
    reduce size by 5%, you increase the chance of a matching size with the model for detection is found. 
    
    This also means that the algorithm works slower since it is more thorough. 
    You may increase it to as much as 1.4 for faster detection, with the risk of missing some faces altogether.

[*] minNeighbors 
    – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    
    This parameter will affect the quality of the detected faces. 
    Higher value results in less detections but with higher quality. 
    3~6 is a good value for it.
    
    Haar cascade classifier works with a sliding window approach. 
    If you look at the cascade files you can see a size parameter 
    which usually a pretty small value like 20 20. 
    This is the smallest window that cascade can detect. 
    So by applying a sliding window approach, 
    you slide a window through out the picture than 
    you resize it and search again until you can not resize it further. 
    So with every iteration haar's cascaded classifier true outputs are stored. 
    So when this window is slided in picture resized and slided again; 
    it actually detects many many false positives. 
    You can check what it detects by giving minNeighbors 0. 


[*] minSize 
– Minimum possible object size. Objects smaller than that are ignored.

This parameter determine how small size you want to detect. You decide it! 
Usually, [30, 30] is a good start for face detection.

[*] maxSize 
– Maximum possible object size. Objects bigger than this are ignored.
This parameter determine how big size you want to detect. 
Again, you decide it! Usually, you don't need to set it manually, 
the default value assumes you want to detect without an upper limit on the size of the face.



'''

hand4 = hand4_cascade.detectMultiScale(
    gray, # Target the region
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(20, 20),
    maxSize=(60, 90),
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    ) 

location

# test
hand1 = hand1_cascade.detectMultiScale(closed) 
hand2 = hand2_cascade.detectMultiScale(closed) 
hand3 = hand3_cascade.detectMultiScale(closed) 

# Size example
leukocytes = leukocyteCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=10,
    minSize=(30, 20),
    maxSize=(60, 90),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

# Show
global img = vid.get_data(1)
def testing_result(obj):
    '''obj = hand2_cascade.detectMultiScale(closed) '''
    a = img.copy
    for (x,y,w,h) in obj:
        # draw rectangle at the specific location
        cv2.rectangle(a,(x,y),(x+w,y+h),(255,0,0),2) 
        # extract the segmentation 
    io.imshow(a)





for (x,y,w,h) in hand3:
    # draw rectangle at the specific location
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) 
    # extract the segmentation 
    roi_gray = gray[y:y+h, x:x+w]


'''
SECTION END
'''

'''
For Img Saving
'''
io.imsave('bg001.jpg',vid.get_data(1000))



faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]

sudo apt-get update
sudo apt-get upgrade
sudo rpi-update
sudo apt-get install build-essential cmake pkg-config