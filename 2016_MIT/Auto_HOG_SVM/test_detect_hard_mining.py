
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

def test_with_pro(frame_id,pro):
    ''' Use this fuc when ### above are first implement'''
    img = vid.get_data(frame_id)
    img = auto_resized(img,conf['train_resolution'])
    img_gray = cv2.cvtColor( img , cv2.COLOR_BGR2GRAY)
    roi = img_gray[20:180,35:345]
    (boxes, probs) = od.detect(roi, winStep=conf["step_size"], winDim=conf["sliding_size"],
    pyramidScale=1.1, minProb=pro)
    # since for size effect with the camera, pyramidScale = 1.001, mnust>1, 
    # if positive size would change, we have to use 1.5 or 2 ...etc 
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    orig = img.copy()
    # loop over the allowed bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(orig, (startX+35, startY+20), (endX+35, endY+20), (0, 255, 0), 2)
        print (startX+35, startY+20), (endX+35, endY+20)
    show(orig)
    return img_gray, orig

# Use like below
# a,b = test_with_pro

def hard_neg_extrect(conf,hn):
    '''nh = img , self crop the hn zone'''
    img = auto_resized(hn ,conf['sliding_size'])
    from datetime import datetime
    from skimage.feature import hog as _hog
    fd=_hog(img, conf['orientations'], conf['pixels_per_cell'], 
                conf['cells_per_block'], conf['visualize'], conf['normalize'])
    
    fd_name = str(datetime.now()).split(':')[0].replace(' ','').replace('-','')
    fd_path = os.path.join(conf['neg_feat_ph'], fd_name)
    for i in range(50):
        joblib.dump(fd, fd_path+str(i)+'.feat')
    print '[*] saving hn-feat to neg_feat_ph, be sure re-train the model'

def test_with_hc(frame_id,pro):
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
        if startY < 50 and startX < 50:
            pass
        elif abs(startX - endX)>80: # Size effect
            pass
        elif (check_hsv2(img)[startY+20:endY+20,startX+35:endX+35]).sum()<255*350:
            # White region = 255
            print (startX, startY, endX, endY)
            print '[*] White Value Index must > 20'
            print (check_hsv2(img)[startY+20:endY+20,startX+35:endX+35]).sum()
            print '[*] End'
        else: 
            cv2.rectangle(orig, (startX+35, startY+20), (endX+35, endY+20), (0, 255, 0), 2)
            print '[*** START ***]'
            print (startX+35, startY+20), (endX+35, endY+20)
            print (check_hsv2(img)[startY+20:endY+20,startX+35:endX+35]).sum()
            print '[*** END ***]'
    show(orig)
    return img_gray, orig

def test_with_hc_222(frame_id,pro):
    ''' Use this fuc when ### above are first implement'''
    img = vid.get_data(frame_id)
    img = auto_resized(img,conf['train_resolution'])
    # ROI roi
    roi = img[20:180,35:345]
    # use check_hsv
    img_gray = check_hsv(roi)
    (boxes, probs) = od.detect(img_gray, winStep=conf["step_size"], winDim=conf["sliding_size"],
    pyramidScale=1.1, minProb=pro)
    # since for size effect with the camera, pyramidScale = 1.001, mnust>1, 
    # if positive size would change, we have to use 1.5 or 2 ...etc 
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    orig = img.copy()
    # loop over the allowed bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        if startY < 50 and startX < 50:
            pass
        elif abs(startX - endX)>80: # Size effect
            pass
        elif (check_hsv2(img)[startY+20:endY+20,startX+35:endX+35]).sum()<255*300:
            # White region = 255
            print (startX, startY, endX, endY)
            print '[*] White Value Index must > 20'
            print (check_hsv2(img)[startY+20:endY+20,startX+35:endX+35]).sum()
            print '[*] End'
        else: 
            cv2.rectangle(orig, (startX+35, startY+20), (endX+35, endY+20), (0, 255, 0), 2)
    show(orig)
    return img_gray, orig

def check_hsv(frame):
    '''
    in : raw RGB frame
    out : GRAY ( white + blur + erode + dilate )
    Note this GRAY img is useful to have Further Use like HAAR ...etc
    '''
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #hsv = cv2.cvtColor(frame, cv2.COLOR_GRAY2HSV)
    lower_white = np.array([0,0,100])
    upper_white = np.array([200,50,300])
    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    gray = cv2.cvtColor(mask, cv2.COLOR_BAYER_GB2GRAY)
    # blur and threshold the image
    blurred=cv2.blur(gray,(9,9))
    (_, thresh) = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)
    return closed

def check_hsv2(frame):
    '''
    in : raw RGB frame
    out : GRAY ( white + blur + erode + dilate )
    Note this GRAY img is useful to have Further Use like HAAR ...etc
    '''
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #hsv = cv2.cvtColor(frame, cv2.COLOR_GRAY2HSV)
    lower_white = np.array([0,0,100])
    upper_white = np.array([200,50,300])
    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    gray = cv2.cvtColor(mask, cv2.COLOR_BAYER_GB2GRAY)
    # blur and threshold the image
    #blurred=cv2.blur(gray,(9,9))
    (_, thresh) = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    #closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # perform a series of erosions and dilations
    #closed = cv2.erode(closed, None, iterations = 4)
    #closed = cv2.dilate(closed, None, iterations = 4)
    return thresh #closed

def test_with_hc_v2(frame_id,pro,falsePositive=False,falseNegative=False):
    ''' Use this fuc when ### above are first implement'''
    img = vid.get_data(frame_id)
    img = auto_resized(img,conf['train_resolution'])
    img_gray = cv2.cvtColor( img , cv2.COLOR_BGR2GRAY)
    roi = img_gray[20:180,35:345]
    (boxes, probs) = od.detect(roi, winStep=conf["step_size"], winDim=conf["sliding_size"],
    pyramidScale=1.1, minProb=pro)
    # since for size effect with the camera, pyramidScale = 1.001, mnust>1, 
    # if positive size would change, we have to use 1.5 or 2 ...etc 
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    orig = img.copy()
    # loop over the allowed bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        if startY < 50 and startX < 50:
            pass
        else : 
            cv2.rectangle(orig, (startX+35, startY+20), (endX+35, endY+20), (0, 255, 0), 2)
        print (startX+35, startY+20), (endX+35, endY+20)
        ###
        ##
        ###
        if falsePositive == True:
            hard_neg_extrect(conf,)
    show(orig)

    return img_gray, orig


box = []
def test_with_hc_pick(frame_id,pro):
    ''' Use this fuc when ### above are first implement'''
    img = vid.get_data(frame_id)
    img = auto_resized(img,conf['train_resolution'])
    img_gray = cv2.cvtColor( img , cv2.COLOR_BGR2GRAY)
    roi = img_gray[20:180,35:345]
    (boxes, probs) = od.detect(roi, winStep=conf["step_size"], winDim=conf["sliding_size"],
    pyramidScale=1.1, minProb=pro)
    # since for size effect with the camera, pyramidScale = 1.001, mnust>1, 
    # if positive size would change, we have to use 1.5 or 2 ...etc 
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    orig = img.copy()
    # loop over the allowed bounding boxes and draw them
    box=[]
    for (startX, startY, endX, endY) in pick:
        if startY < 50 and startX < 50:
            pass
        else : 
            cv2.rectangle(orig, (startX+35, startY+20), (endX+35, endY+20), (0, 255, 0), 2)
            print (startX+35, startY+20, endX+35, endY+20)
            box +=[startX+35, startY+20, endX+35, endY+20]
    return box

# export motion
box = []
for i in range(12000):
    box+=test_with_hc_pick(i,0.3)



# explore motion
box = []
for i in range(100):
    box+=[test_with_hc_pick(i*5,0.3)]
    test_with_hc(i*5,0.3)





def get_post_sample(frame_id):
    ''' Use this fuc when ### above are first implement'''
    img = vid.get_data(frame_id)
    img = auto_resized(img,conf['train_resolution'])
    show(img)
    return img

def save_to_pos_raw(conf,img,name):
    'name = str(.png)'
    from skimage.io import imsave
    img = auto_resized(img,conf['train_size'])
    fd_path = os.path.join(conf['pos_raw_ph'], name)
    assert fd_path.split('.')[1]=='png'
    imsave(fd_path,img)
    print 'Save Target to' + fd_path
    # check the dimenstion

def crop_and_save_neg_raw(conf,img,name,box):
    ''' box = starX,startY,endX,endY'''
    # Use Bigger Box => make target small
    starX,startY,endX,endY = box  
    from skimage.io import imsave
    img = auto_resized(img,conf['train_resolution'])
    img = img[startY:endY,startX:endX,:]
    fd_path = os.path.join(conf['neg_raw_ph'], name)
    assert fd_path.split('.')[1]=='png'
    #show(img)
    imsave(fd_path,img)
    print 'Save Target to' + fd_path

def crop_and_save_pos_raw(conf,img,name,box):
    ''' box = starX,startY,endX,endY'''
    # Use Bigger Box => make target small
    starX,startY,endX,endY = box  
    from skimage.io import imsave
    img = auto_resized(img,conf['train_resolution'])
    img = img[startY:endY,startX:endX,:]
    fd_path = os.path.join(conf['pos_raw_ph'], name)
    assert fd_path.split('.')[1]=='png'
    #show(img)
    imsave(fd_path,img)
    print 'Save Target to' + fd_path


for i in [2,200,190,191,291,292,251,951,952,953,1953,1952,1962,2162,2163,2164,2165,2265,2366]:
    name = 'sd_scale_'+str(i)+'.png'
    crop_and_save_pos_raw(conf,vid.get_data(i),name,box2)



imgg[83:123,190:230,:]







##############

# pro = 0.11 with first 
# raw_img = vid.get_data(1)