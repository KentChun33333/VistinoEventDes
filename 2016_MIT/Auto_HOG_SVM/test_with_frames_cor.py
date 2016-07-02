
def main_testing():
    '''cd /Users/kentchiu/Night_Graden/Project/2016_MIT/Auto_HOG_SVM/'''
    from common_tool_agent.common_func import non_max_suppression
    from common_tool_agent.conf import Conf
    from common_tool_agent.descriptor_agent.hog import HOG
    from common_tool_agent.detect import ObjectDetector
    from sklearn.externals import joblib
    from skimage.io import imread
    from skimage.io import imshow as show
    from common_tool_agent.common_func import auto_resized
    import argparse 
    import numpy as np
    import cv2
    import os
    import imageio
    conf = Conf('conf_hub/conf_001.json')
    #conf = Conf('conf_hub/conf_001.json')
    vid = imageio.get_reader('/Users/kentchiu/MIT_Vedio/2016-01-21/10.167.10.158_01_20160121082638418_1.mp4')
    # loading model
    clf = joblib.load(conf['model_ph'])

    # initialize feature container
    hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
    cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
    # initialize the object detector
    od = ObjectDetector(clf, hog)

# A/B Test

frame_id=0
frame_que=[]
hand_position = []
frame_frame_correlation = []
for i in range(1130):
    # init
    img = vid.get_data(i*3)
    hand,frame =cascade_test_15(img)
    hand_position.append(hand)
    frame_que.append(frame)
    frame_id+=1
    if frame_id >=15:
        hand_temp = hand_position[frame_id-15:frame_id-1]


# Save
output = 'output.mp4'
height, width, channels = frame_que[0].shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 3.0, (width, height))

for image in frame_que:
    out.write(image) # Write out frame to video

# Release everything if job is finished
out.release()

cv2.destroyAllWindows()

print ("The output video is {}".format(output))


def test_video_saving():
    frame_id=0
    frame_que=[]
    hand_position = []
    frame_frame_correlation = []
    for i in range(1130):
        # init
        img = vid.get_data(i*3)
        hand,frame =cascade_test_15(img)
        hand_position.append(hand)
        frame_que.append(frame)
        frame_id+=1
        if frame_id >=15:
            hand_temp = hand_position[frame_id-15:frame_id-1]




#===========================

def context_aware_speed(ref_pt,new_pt,speed):
	'''pt = turple(x,y)'''
	x1,y1 = ref_pt
	x2,y2 = new_pt
	dist = sqrt( (x2 - x1)**2 + (y2 - y1)**2)
	if dist > speed:
		return ref_pt
	else:
		return new_pt

def context_aware_speed_2(ref_pt_set,new_pt_set,speed):
	'''pt_Set = list[turple(x,y)]'''

	if ref_pt_set is not None:
		x1,y1 = ref_pt_set.pop()
		ith = 0
		if new_pt_set is not None:
			x2,y2 = new_pt_set.pop()
			dist = sqrt( (x2 - x1)**2 + (y2 - y1)**2)
	if dist > speed:
		return ref_pt
	else:
		return new_pt


import pandas as pd
def context_aware_two_hands(ref_pt_set, new_pt_set):
	'''chose the close one'''
	init=[]
	for i in range(len(new_pt_set)):
		init.append([0]*len(ref_pt_set))

	memo = np.array(init)
	for i in range(len(ref_pt_set)):
		for j in range(len(new_pt_set)):
			x1,y1 = ref_pt_set[i]
			x2,y2 = new_pt_set[]j
			dist = sqrt( (x2 - x1)**2 + (y2 - y1)**2)
			memo[j][1]=dist
			memo[]][0]=j
	memo = pd.DataFrame(memo)
	memo = memo.sort_values(by=1)
	memo = list(memo[0])
	tmp = memo.pop(0)
	tmp2 = memo.pop(0)
	while tmp==tmp2:
		tmp2 = memo.pop(0)
	result = [new_pt_set[tmp],new_pt_set[tmp2]]
	return result

#===========

# Algo2




def cascade_test_15(img):
    hand_5 = cv2.CascadeClassifier(
    '/Users/kentchiu/MIT_Vedio/Rhand_no_tools/cascade.xml')
    ref = img.copy()
    hand5 = hand_5.detectMultiScale(
        ref.copy()[50:550,50:800],
        scaleFactor=1.2,
        minNeighbors=15, #35 ==> 1
        #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    hand = []
    for (x,y,w,h) in hand5:
        # draw rectangle at the specific location
        cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 
        hand.append([(x+50+x+50+w)/2,(y+50+h+y+50)/2])
    #show(ref)

        print ((x+50,y+50),(x+50+w,y+50+h))
    return hand, ref
# Testing 9
def cascade_test_15(img):
    hand_5 = cv2.CascadeClassifier(
    '/Users/kentchiu/MIT_Vedio/Rhand_no_tools/cascade.xml')
    ref = img.copy()
    hand5 = hand_5.detectMultiScale(
        ref.copy()[50:550,50:800],
        scaleFactor=1.2,
        minNeighbors=15, #35 ==> 1
        #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    hand = []
    for (x,y,w,h) in hand5:
        if x < 160 or y < 160 or h < 100:
            pass
        else:
        # draw rectangle at the specific location
            cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 
            hand.append([(x+50+x+50+w)/2,(y+50+h+y+50)/2])
    #show(ref)
            print ((x+50,y+50),(x+50+w,y+50+h))
    return hand, ref


# Testing 2016-May24-19:30
def cascade_test_15_c(img):
    hand_5 = cv2.CascadeClassifier(
    '/Users/kentchiu/MIT_Vedio/Rhand_no_tools/cascade.xml')
    ref = cv2.GaussianBlur(img,(5,5),0)
    hand5 = hand_5.detectMultiScale(
        ref.copy()[50:550,50:800],
        scaleFactor=1.2,
        minNeighbors=15, #35 ==> 1
        #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    hand = []
    for (x,y,w,h) in hand5:
    	# Position Aware + Size Aware
        if x < 160 or y < 160 or h<90:
            pass
        else:
        # draw rectangle at the specific location
            cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 
            hand.append([(x+50+x+50+w)/2,(y+50+h+y+50)/2])
    #show(ref)
            print ((x+50,y+50),(x+50+w,y+50+h))
    return hand, ref

# Sudo Code

if new_frame_position_set.length > 2 

filter to 2 


def cascade_test_15_c2(img):
    hand_5 = cv2.CascadeClassifier(
    '/Users/kentchiu/MIT_Vedio/Rhand_no_tools/cascade.xml')
    ref = cv2.GaussianBlur(img,(5,5),0)
    hand5 = hand_5.detectMultiScale(
        ref.copy()[50:550,50:800],
        scaleFactor=1.2,
        minNeighbors=15, #35 ==> 1
        #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    hand = []
    for (x,y,w,h) in hand5:
    # Position Aware + Size Aware
        if x < 160 or y < 160 or h<90:
            pass
        else:
        # draw rectangle at the specific location
            cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 
            hand.append([(x+50+x+50+w)/2,(y+50+h+y+50)/2])
    #show(ref)
            print ((x+50,y+50),(x+50+w,y+50+h))
    return hand

frame_id = []
hand_temp = []
for i in range(1130):
    # init
    img = vid.get_data(i*3)
    frame_id.append(i*3)
    hand =cascade_test_15_c2(img)
    hand_temp.append(hand)

#######
#######

def cascade_test_15_c3(img):
    hand_5 = cv2.CascadeClassifier(
    '/Users/kentchiu/MIT_Vedio/Rhand_no_tools/cascade.xml')
    ref = img.copy()
    hand5 = hand_5.detectMultiScale(
        ref.copy()[50:550,50:800],
        scaleFactor=1.2,
        minNeighbors=15, #35 ==> 1
        #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    hand = []
    for (x,y,w,h) in hand5:
    # Position Aware + Size Aware
        if (x < 160 and y < 160) or y<160 or h<90:
            pass
        else:
        # draw rectangle at the specific location
            cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 
            hand.append([(x+50+x+50+w)/2,(y+50+h+y+50)/2])
    #show(ref)
            print ((x+50,y+50),(x+50+w,y+50+h))
    return ref



frame_id = []
hand_temp = []
for i in range(630):
    # init
    img = vid.get_data(i*3)
    frame_id.append(i*3)
    hand =cascade_test_15_c3(img)
    hand_temp.append(hand)


output = 'output_c3_2.mp4'
height, width, channels = vid.get_data(0).shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
#fourcc = cv2.VideoWriter_fourcc(*'X264') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 6.0, (width, height))

for image in hand_temp:
    out.write(image) # Write out frame to video

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()


###########
######### #
##      # #
######### #
###########



def cascade_test_15_c4(img):
    hand_5 = cv2.CascadeClassifier(
    '/Users/kentchiu/MIT_Vedio/Rhand_no_tools/cascade.xml')
    ref = cv2.GaussianBlur(img,(5,5),0)
    hand5 = hand_5.detectMultiScale(
        ref.copy()[50:550,50:800],
        scaleFactor=1.2,
        minNeighbors=12, #35 ==> 1
        #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    hand = []
    for (x,y,w,h) in hand5:
    # Position Aware + Size Aware
        if (x < 160 and y < 160) or y<160 or h<90:
            pass
        else:
        # draw rectangle at the specific location
            cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 
            hand.append([(x+50+x+50+w)/2,(y+50+h+y+50)/2])
    #show(ref)
            print ((x+50,y+50),(x+50+w,y+50+h))
    return ref



frame_id = []
hand_temp = []
for i in range(630):
    # init
    img = vid.get_data(i*3)
    frame_id.append(i*3)
    hand =cascade_test_15_c3(img)
    hand_temp.append(hand)


output = 'output_c4.mp4'
height, width, channels = vid.get_data(0).shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 6.0, (width, height))

for image in hand_temp:
    out.write(image) # Write out frame to video

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()



###########
######### #
##      # #
######### #
###########

# 2016-May25
def test_with_hc(frame_id,pro=0.3):
    ''' Use this fuc when ### above are first implement'''
    img = vid.get_data(frame_id)
    img = auto_resized(img,conf['train_resolution'])
    # ROI roi
    roi = img[20:180,35:345]
    img_gray = cv2.cvtColor( roi , cv2.COLOR_BGR2GRAY)
    (boxes, probs) = od.detect(img_gray, winStep=conf["step_size"], winDim=conf["sliding_size"],
    pyramidScale=1.1, minProb=pro)
    # since for size effect with the camera, pyramidScale = 1.001, mnust>1, 
    # if positive size would change, we have to use 1.5 or 2 ...etc 
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    orig = img.copy()
    # loop over the allowed bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        if (startY < 50 and startX < 50) or abs(startX - endX)>80 or abs(startX - endX)<30 :
            pass
        else: 
            cv2.rectangle(orig, (startX+35, startY+20), (endX+35, endY+20), (0, 255, 0), 2)
    return orig


frame_id = []
hand_temp = []
for i in range(530):
    # init
    hand =test_with_hc(i*3)
    hand_temp.append(hand)


output = 'output_test_with_hc.mp4'
height, width, channels = hand_temp[0].shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 6.0, (width, height))

for image in hand_temp:
    out.write(image) # Write out frame to video

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

#=====================================================================================


def main_testing():
    '''cd /Users/kentchiu/Night_Graden/Project/2016_MIT/Auto_HOG_SVM/'''
    from common_tool_agent.common_func import non_max_suppression
    from common_tool_agent.conf import Conf
    from common_tool_agent.descriptor_agent.hog import HOG
    from common_tool_agent.detect import ObjectDetector
    from sklearn.externals import joblib
    from skimage.io import imread
    from skimage.io import imshow as show
    from common_tool_agent.common_func import auto_resized
    import argparse 
    import numpy as np
    import cv2
    import os
    import imageio
    conf = Conf('conf_hub/conf_Rhand_3.json')
    #conf = Conf('conf_hub/conf_001.json')
    vid = imageio.get_reader('/Users/kentchiu/MIT_Vedio/2016-01-21/10.167.10.158_01_20160121082638418_1.mp4')
    # loading model
    clf = joblib.load(conf['model_ph'])

    # initialize feature container
    hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
    cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
    # initialize the object detector
    od = ObjectDetector(clf, hog)

# 2016-May25
def test_with_hc(frame_id,pro=0.3):
    ''' Use this fuc when ### above are first implement'''
    img = vid.get_data(frame_id)
    img = auto_resized(img,conf['train_resolution'])
    # ROI roi
    roi = img[20:180,35:345]
    img_gray = cv2.cvtColor( roi , cv2.COLOR_BGR2GRAY)
    (boxes, probs) = od.detect(img_gray, winStep=conf["step_size"], winDim=conf["sliding_size"],
    pyramidScale=1.1, minProb=pro)
    # since for size effect with the camera, pyramidScale = 1.001, mnust>1, 
    # if positive size would change, we have to use 1.5 or 2 ...etc 
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    orig = img.copy()
    # loop over the allowed bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        if (startY < 50 and startX < 50) or abs(startX - endX)>80 or abs(startX - endX)<30 :
            pass
        else: 
            cv2.rectangle(orig, (startX+35, startY+20), (endX+35, endY+20), (0, 255, 0), 2)
    return orig


frame_id = []
hand_temp = []
for i in range(530):
    # init
    hand =test_with_hc(i*3)
    hand_temp.append(hand)


output = 'Conf_Rhand_3.mp4'
height, width, channels = hand_temp[0].shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 6.0, (width, height))

for image in hand_temp:
    out.write(image) # Write out frame to video

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()




#=====================================================================================




def main_testing():
    '''cd /Users/kentchiu/Night_Graden/Project/2016_MIT/Auto_HOG_SVM/'''
    from common_tool_agent.common_func import non_max_suppression
    from common_tool_agent.conf import Conf
    from common_tool_agent.descriptor_agent.hog import HOG
    from common_tool_agent.detect import ObjectDetector
    from sklearn.externals import joblib
    from skimage.io import imread
    from skimage.io import imshow as show
    from common_tool_agent.common_func import auto_resized
    import argparse 
    import numpy as np
    import cv2
    import os
    import imageio
    conf = Conf('conf_hub/conf_Rhand_4.json')
    #conf = Conf('conf_hub/conf_001.json')
    vid = imageio.get_reader('/Users/kentchiu/MIT_Vedio/2016-01-21/10.167.10.158_01_20160121082638418_1.mp4')
    # loading model
    clf = joblib.load(conf['model_ph'])

    # initialize feature container
    hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
    cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
    # initialize the object detector
    od = ObjectDetector(clf, hog)

# 2016-May25
def test_with_hc(frame_id,pro=0.3):
    ''' Use this fuc when ### above are first implement'''
    img = vid.get_data(frame_id)
    img = auto_resized(img,conf['train_resolution'])
    # ROI roi
    roi = img[20:180,35:345]
    img_gray = cv2.cvtColor( roi , cv2.COLOR_BGR2GRAY)
    (boxes, probs) = od.detect(img_gray, winStep=conf["step_size"], winDim=conf["sliding_size"],
    pyramidScale=1.1, minProb=pro)
    # since for size effect with the camera, pyramidScale = 1.001, mnust>1, 
    # if positive size would change, we have to use 1.5 or 2 ...etc 
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    orig = img.copy()
    # loop over the allowed bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        if (startY < 50 and startX < 50) or abs(startX - endX)>80 or abs(startX - endX)<30 :
            pass
        else: 
            cv2.rectangle(orig, (startX+35, startY+20), (endX+35, endY+20), (0, 255, 0), 2)
    return orig


frame_id = []
hand_temp = []
for i in range(530):
    # init
    hand =test_with_hc(i*3)
    hand_temp.append(hand)


output = 'conf_Rhand_4.mp4'
height, width, channels = hand_temp[0].shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 6.0, (width, height))

for image in hand_temp:
    out.write(image) # Write out frame to video

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()






#=====================================================================================




def main_testing():
    '''cd /Users/kentchiu/Night_Graden/Project/2016_MIT/Auto_HOG_SVM/'''
    from common_tool_agent.common_func import non_max_suppression
    from common_tool_agent.conf import Conf
    from common_tool_agent.descriptor_agent.hog import HOG
    from common_tool_agent.detect import ObjectDetector
    from sklearn.externals import joblib
    from skimage.io import imread
    from skimage.io import imshow as show
    from common_tool_agent.common_func import auto_resized
    import argparse 
    import numpy as np
    import cv2
    import os
    import imageio
    conf = Conf('conf_hub/conf_Rhand_7.json')
    #conf = Conf('conf_hub/conf_001.json')
    vid = imageio.get_reader('/Users/kentchiu/MIT_Vedio/2016-01-21/10.167.10.158_01_20160121082638418_1.mp4')
    # loading model
    clf = joblib.load(conf['model_ph'])

    # initialize feature container
    hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
    cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
    # initialize the object detector
    od = ObjectDetector(clf, hog)

def test_with_hc(frame_id,pro=0.3):
    ''' Use this fuc when ### above are first implement'''
    img = vid.get_data(frame_id)
    img = auto_resized(img,conf['train_resolution'])
    # ROI roi
    roi = img[20:180,35:345]
    img_gray = cv2.cvtColor( roi , cv2.COLOR_BGR2GRAY)
    (boxes, probs) = od.detect(img_gray, winStep=conf["step_size"], winDim=conf["sliding_size"],
    pyramidScale=1.1, minProb=pro)
    # since for size effect with the camera, pyramidScale = 1.001, mnust>1, 
    # if positive size would change, we have to use 1.5 or 2 ...etc 
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    orig = img.copy()
    # loop over the allowed bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        if (startY < 50 and startX < 50) or abs(startX - endX)>80 or abs(startX - endX)<30 :
            pass
        else: 
            cv2.rectangle(orig, (startX+35, startY+20), (endX+35, endY+20), (0, 255, 0), 2)
    return orig

def test_with_hc_222(frame_id,pro=0.32):
    ''' Use this fuc when ### above are first implement'''
    img = vid.get_data(frame_id)
    img = auto_resized(img,conf['train_resolution'])
    # ROI roi
    roi = img[20:180,35:345]
    # use check_hsv
    img_gray = cv2.cvtColor( roi , cv2.COLOR_BGR2GRAY)
    (boxes, probs) = od.detect(img_gray, winStep=conf["step_size"], winDim=conf["sliding_size"],
    pyramidScale=1.1, minProb=pro)
    # since for size effect with the camera, pyramidScale = 1.001, mnust>1, 
    # if positive size would change, we have to use 1.5 or 2 ...etc 
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    orig = img.copy()
    # loop over the allowed bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        wid = endX-startX
        if (startY < 50 and startX < 50) or abs(startX - endX)>80 or abs(startX - endX)<20 :
            pass
        elif (check_hsv(orig)[startY+20+0.25*wid:startY+20+0.75*wid,startX+20+0.25*wid:startX+20+0.75*wid]).sum()<255*(endX-startX)/2:
            # White region = 255
            print (startX, startY, endX, endY)
            print (check_hsv(orig)[startY+20+0.25*wid:startY+20+0.75*wid,startX+20+0.25*wid:startX+20+0.75*wid]).sum()
            pass
            print '[*]'
        else: 
            cv2.rectangle(orig, (startX+35, startY+20), (endX+35, endY+20), (0, 255, 0), 2)
            print (startX, startY, endX, endY)
            print (check_hsv(orig)[startY+20+0.25*wid:startY+20+0.75*wid,startX+20+0.25*wid:startX+20+0.75*wid]).sum()
    return orig

def check_hsv(frame):
    '''
    in : raw RGB frame
    out : GRAY ( white + blur + erode + dilate )
    Note this GRAY img is useful to have Further Use like HAAR ...etc
    '''
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0,0,200])
    upper_white = np.array([150,50,300])
    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    gray = cv2.cvtColor(mask, cv2.COLOR_BAYER_GB2GRAY)
    #####################################
    # blur and threshold the image
    # blur size associated with resolution
    ######################################
    blurred=cv2.blur(gray,(3,3))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)
    return closed

frame_id = []
hand_temp = []
for i in range(530):
    # init
    hand =test_with_hc_222(i*3)
    hand_temp.append(hand)


output = 'conf_Rhand_7_white_bb.mp4'
height, width, channels = hand_temp[0].shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 5.0, (width, height))

for image in hand_temp:
    out.write(image) # Write out frame to video

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

#=======================================================================




conf = Conf('conf_hub/conf_Rhand_with_tool_2.json')
#conf = Conf('conf_hub/conf_001.json')
vid = imageio.get_reader('/Users/kentchiu/MIT_Vedio/2016-01-21/10.167.10.158_01_20160121082638418_1.mp4')
# loading model
clf = joblib.load(conf['model_ph'])

# initialize feature container
hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
# initialize the object detector
od = ObjectDetector(clf, hog)


def test_detect(frame_id,pro=0.3):
    ''' Use this fuc when ### above are first implement'''
    img = vid.get_data(frame_id)
    img = auto_resized(img,conf['train_resolution'])
    # ROI roi
    roi = img[20:180,35:345]
    img_gray = cv2.cvtColor( roi , cv2.COLOR_BGR2GRAY)
    (boxes, probs) = od.detect(img_gray, winStep=conf["step_size"], winDim=conf["sliding_size"],
    pyramidScale=1.1, minProb=pro)
    # since for size effect with the camera, pyramidScale = 1.001, mnust>1, 
    # if positive size would change, we have to use 1.5 or 2 ...etc 
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    orig = img.copy()
    # loop over the allowed bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        if (startY < 50 and startX < 50) or abs(startX - endX)>55 or abs(startX - endX)<30 :
            pass
        else: 
            cv2.rectangle(orig, (startX+35, startY+20), (endX+35, endY+20), (155, 155, 0), 2)
    return orig


frame_id = []
hand_temp = []
for i in range(530):
    # init
    hand =test_detect(i*3)
    hand_temp.append(hand)

###############################


output = 'Rhand_with_tool_2.mp4'
height, width, channels = hand_temp[0].shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 5.0, (width, height))

for image in hand_temp:
    out.write(image) # Write out frame to video

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

####################

frame_id = []
hand_temp = []
for i in range(530):
    # init
    hand =test_detect(i*3+300,0.5)
    hand_temp.append(hand)



output = 'Rhand_with_tool_2b.mp4'
height, width, channels = hand_temp[0].shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 5.0, (width, height))

for image in hand_temp:
    out.write(image) # Write out frame to video

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()



#=======================================================================


conf = Conf('conf_hub/conf_Rhand_with_tool_3.json')
#conf = Conf('conf_hub/conf_001.json')
vid = imageio.get_reader('/Users/kentchiu/MIT_Vedio/2016-01-21/10.167.10.158_01_20160121082638418_1.mp4')
# loading model
clf = joblib.load(conf['model_ph'])

# initialize feature container
hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
# initialize the object detector
od = ObjectDetector(clf, hog)


def test_detect(frame_id,pro=0.3):
    ''' Use this fuc when ### above are first implement'''
    img = vid.get_data(frame_id)
    img = auto_resized(img,conf['train_resolution'])
    # ROI roi
    roi = img[20:180,35:345]
    img_gray = cv2.cvtColor( roi , cv2.COLOR_BGR2GRAY)
    (boxes, probs) = od.detect(img_gray, winStep=conf["step_size"], winDim=conf["sliding_size"],
    pyramidScale=1.1, minProb=pro)
    # since for size effect with the camera, pyramidScale = 1.001, mnust>1, 
    # if positive size would change, we have to use 1.5 or 2 ...etc 
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    orig = img.copy()
    # loop over the allowed bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        if (startY < 50 and startX < 50) or abs(startX - endX)>55 or abs(startX - endX)<30 :
            pass
        else: 
            cv2.rectangle(orig, (startX+35, startY+20), (endX+35, endY+20), (155, 155, 0), 2)
    return orig


frame_id = []
hand_temp = []
for i in range(530):
    # init
    hand =test_detect(i*3+300,0.5)
    hand_temp.append(hand)

###############################


output = 'Rhand_with_tool_3.mp4'
height, width, channels = hand_temp[0].shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 5.0, (width, height))

for image in hand_temp:
    out.write(image) # Write out frame to video

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()


#============== May 27 ====================================================

def cascade_test_15_c3(img):
    hand_5 = cv2.CascadeClassifier(
    '/Users/kentchiu/MIT_Vedio/Rhand_no_tools/cascade.xml')
    ref = img.copy()
    hand5 = hand_5.detectMultiScale(
        ref.copy()[50:550,50:800],
        scaleFactor=1.2,
        minNeighbors=15, #35 ==> 1
        #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    hand = []
    for (x,y,w,h) in hand5:
    # Position Aware + Size Aware
        if (x < 160 and y < 160) or y<160 or h<90:
            pass
        else:
        # draw rectangle at the specific location
            cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 
            hand.append([(x+50+x+50+w)/2,(y+50+h+y+50)/2])
    #show(ref)
            print ((x+50,y+50),(x+50+w,y+50+h))
    return ref



frame_id = []
hand_temp = []
for i in range(630):
    # init
    img = vid.get_data(i*3)
    frame_id.append(i*3)
    hand =cascade_test_15_c3(img)
    hand_temp.append(hand)

#------Saving vedio
output = 'output_c3_Nice.mp4'
height, width, channels = vid.get_data(0).shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
#fourcc = cv2.VideoWriter_fourcc(*'x264') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 5.0, (width, height))

for image in hand_temp:
    out.write(image) # Write out frame to video

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

