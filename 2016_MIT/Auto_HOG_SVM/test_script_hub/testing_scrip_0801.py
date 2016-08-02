


def main_testing_conf():
    '''cd /Users/kentchiu/VistinoEventDes/2016_MIT/Auto_HOG_SVM/'''
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
    
    #conf = Conf('conf_hub/conf_001.json')
    vid = imageio.get_reader('/Users/kentchiu/MIT_Vedio/2016-01-21/10.167.10.158_01_20160121082638418_2.mp4')
    # loading model
    
    conf = Conf('conf_hub/conf_RHand_with_tool_4.json')
    clf = joblib.load(conf['model_ph'])

    # initialize feature container
    hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
    cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
    # initialize the object detector
    od = ObjectDetector(clf, hog)



#-------------------------------------------------------------------------
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
    hand_roi=[]
    for (x,y,w,h) in hand5:
    # Position Aware + Size Aware
        if (x < 160 and y < 160) or y<160 or h<90:
            pass
        else:
        # draw rectangle at the specific location
            cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 
            cv2.putText(ref, "Hand", (x+50,y+50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)
            hand.append([(x+50+x+50+w)/2,(y+50+h+y+50)/2])
    #show(ref)
            print ((x+50,y+50),(x+50+w,y+50+h))
            hand_roi.append([x+50,y+50,x+50+w,y+50+h])
    if len(hand_roi)>1:
        result_box = max(hand_roi)
    elif len(hand_roi)==1:
        result_box = hand_roi.pop()
    else:
        result_box = 0
    return ref, result_box


def sequence_container(inPut, seqLen):
    '''
    1. inPut = list or tuple or .... anything with len(),
    2. seqLen = limit the len, 
    3. Use Scenario : 
    '''
    if len(inPut)>seqLen:
        for i in range(seqLen):
            inPut[i]=inPut[i+1]
        del inPut[-1]
        return inPut
    else :
        return inPut

def template_match(refImg, newImg):
    res = cv2.matchTemplate(newImg, refImg, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    h , w, _ = refImg.shape
    bottom_right = (top_left[0]+w,top_left[1]+h)
    print (top_left, bottom_right)
    cv2.rectangle(newImg, top_left, bottom_right, 255, 2)
    #show(newImg)
    #NewTemplate =newImg[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
    return newImg, (top_left[0],top_left[1], bottom_right[0],bottom_right[1])

def template_match_2(refImg, newImg):
    res = cv2.matchTemplate(newImg, refImg, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    h , w, = refImg.shape
    bottom_right = (top_left[0]+w,top_left[1]+h)
    print (top_left, bottom_right)
    cv2.rectangle(newImg, top_left, bottom_right, 255, 2)
    #show(newImg)
    #NewTemplate =newImg[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
    return newImg, (top_left[0],top_left[1], bottom_right[0],bottom_right[1])

def checkState_0():
    return 1

def checkState_1(memo,seqLen):
    '''memo = memoState_1'''
    if len(memo) < seqLen :
        return 1
    else:
        if memo[-1]==memo[-4] and memo[-4]==memo[-8] and memo[-8]==memo[-12] and memo[-12]== memo[-15]:
            return 0
        else:
            return 1

def video_saving(fileName, fps, imgSequence):
    height, width, channels = imgSequence[0].shape    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(fileName, fourcc, fps, (width, height))    
    for image in imgSequence:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        out.write(image) # Write out frame to video    
    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

def dep_video_saving(fileName, fps, imgSequence):
    height, width = imgSequence[0].shape    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(fileName, fourcc, fps, (width, height))    
    for image in imgSequence:
        img = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB) # avoid 3|4 ERROR
        out.write(img) # Write out frame to video    
    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
#---> in a batch way
#---> conf.

def main_script(newImg,stateID,seqLen):
    #-----init-----
    global memoState_1
    if stateID==0:
        img, tarBox = cascade_test_15_c3(newImg)
        if tarBox==0:
            print '[*] the tarBox is None as recognition'
            result_img.append(img)
            result_position.append(tarBox)
            return
        believe = checkState_0()
        if believe==1:
            print (stateID)
            #-> Output 
            result_img.append(img)
            result_position.append(tarBox)
            #-> Init extra-variable with state 1
            global state 
            state=1 
            global refImg
            refImg = img[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2],:]
            #-> Clean memo_0
            memoState_0=[]
            return 
        else:
            memoState_0.append(img)
            memoState_0 = sequence_container(memoState_0,5)
            return
    else:
        img, tarBox = template_match(refImg, newImg)
        believe = checkState_1(memoState_1,seqLen)
        if believe==1:
            print (stateID)
            #-> Output
            global refImg
            refImg = img[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2],:]
            result_img.append(img)
            result_position.append(tarBox)
            #-> Memo
            memoState_1.append(tarBox)
            memoState_1 = sequence_container(memoState_1,seqLen)
            return
        else:
            global state
            state=0
            #-> Clean memo
            global memoState_1 
            memoState_1=[]
            return




def main(fileName, vid, maxFrameID,seqLen):

memo = []
memoState_0 = []
memoState_1 = []
result_img = []
result_position =[]
state = 0 # variable better not the same as one in the function
seqLen = 15

for i in range(1000):
    main_script(vid.get_data(i),state,seqLen)



video_saving('~/MIT_Vedio/test_0601_1.mp4',20.0,result_img) 

#
#
#
#

def configuration_():
    vid = imageio.get_reader('/Users/kentchiu/MIT_Vedio/2016-01-21/10.167.10.158_01_20160121082638418_2.mp4')
    conf = Conf('conf_hub/conf_pureScrewDriver.json')
    clf = joblib.load(conf['model_ph'])
    # initialize feature container
    hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
    cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
    # initialize the object detector
    od = ObjectDetector(clf, hog)

def configuration_2():
    vid = imageio.get_reader('/Users/kentchiu/MIT_Vedio/2016-01-21/10.167.10.158_01_20160121082638418_2.mp4')
    conf = Conf('conf_hub/conf_pureScrewDriver_2.json')
    clf = joblib.load(conf['model_ph'])
    # initialize feature container
    hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
    cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
    # initialize the object detector
    od = ObjectDetector(clf, hog)

def test_with_pro(rawImg,pro):
    ''' Use this fuc when ### above are first implement'''
    ref = rawImg.copy()
    img = auto_resized(rawImg,conf['train_resolution'])
    img_gray = cv2.cvtColor( img , cv2.COLOR_BGR2GRAY)
    roi = img_gray
    (boxes, probs) = od.detect(roi, winStep=conf["step_size"], winDim=conf["sliding_size"],
    pyramidScale=1.1, minProb=pro)
    # since for size effect with the camera, pyramidScale = 1.001, mnust>1, 
    # if positive size would change, we have to use 1.5 or 2 ...etc 
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    orig = img.copy()

    # Resize Back, I am GOD !!!!! 
    y_sizeFactor = ref.shape[0]/float(img.shape[0])
    x_sizeFactor = ref.shape[1]/float(img.shape[1])

    # loop over the allowed bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        #cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        startX = int(startX* x_sizeFactor)
        endX   = int(endX  * x_sizeFactor)
        startY = int(startY* y_sizeFactor)
        endY   = int(endY  * y_sizeFactor)        
        cv2.rectangle(ref, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(ref, "SkewDriver", (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        #print (startX, startY), (endX, endY)
    #show(ref)
    return ref, pick

def test_with_pro_2(rawImg,pro):
    ''' Use this fuc when ### above are first implement'''
    ref = rawImg.copy()
    img = auto_resized(rawImg,conf['train_resolution'])
    img_gray = cv2.cvtColor( img , cv2.COLOR_BGR2GRAY)
    roi = img_gray
    (boxes, probs) = od.detect(roi, winStep=conf["step_size"], winDim=conf["sliding_size"],
    pyramidScale=1.1, minProb=pro)
    # since for size effect with the camera, pyramidScale = 1.001, mnust>1, 
    # if positive size would change, we have to use 1.5 or 2 ...etc 
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    orig = img.copy()

    # Resize Back, I am GOD !!!!! 
    y_sizeFactor = ref.shape[0]/float(img.shape[0])
    x_sizeFactor = ref.shape[1]/float(img.shape[1])

    # loop over the allowed bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        #cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        startX = int(startX* x_sizeFactor)
        endX   = int(endX  * x_sizeFactor)
        startY = int(startY* y_sizeFactor)
        endY   = int(endY  * y_sizeFactor)        
        cv2.rectangle(ref, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(ref, "Hodling SkrewDriver", (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        #print (startX, startY), (endX, endY)
    show(ref)
    return ref, pick

def main_script(newImg,stateID,seqLen):
    #-----init-----
    global memoState_1
    if stateID==0:
        #img, tarBox = cascade_test_15_c3(newImg)
        img,tarBox = test_with_pro(newImg,0.9)
        if tarBox==0:
            print '[*] the tarBox is None as recognition'
            result_img.append(img)
            result_position.append(tarBox)
            return
        believe = checkState_0()
        if believe==1:
            print (stateID)
            #-> Output 
            result_img.append(img)
            result_position.append(tarBox)
            #-> Init extra-variable with state 1
            global state 
            state=1 
            global refImg
            refImg = img[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2],:]
            #-> Clean memo_0
            memoState_0=[]
            return 
        else:
            memoState_0.append(img)
            memoState_0 = sequence_container(memoState_0,5)
            return
    else:
        img, tarBox = template_match(refImg, newImg)
        believe = checkState_1(memoState_1,seqLen)
        if believe==1:
            print (stateID)
            #-> Output
            global refImg
            refImg = img[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2],:]
            result_img.append(img)
            result_position.append(tarBox)
            #-> Memo
            memoState_1.append(tarBox)
            memoState_1 = sequence_container(memoState_1,seqLen)
            return
        else:
            global state
            state=0
            #-> Clean memo
            global memoState_1 
            memoState_1=[]
            return

memo = []
memoState_0 = []
memoState_1 = []
result_img = []
result_position =[]
state = 0 # variable better not the same as one in the function
seqLen = 15

for i in range(1000):
    main_script(vid.get_data(i),state,seqLen)
