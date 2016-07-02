#-------------------------------------------------------------
# Author : Kent Chiu   
#-------------------------------------------------------------
# Description : 
# 
# This py is motion_tracking module with recognisor 
# there are 2 states 
# 1 states is for detecting
# 1 states is for near neighbor Template Match
# there are 2 lenth limit (FIFO) Que (function) as DP memo
# 
# This is a testing and intuitive script 
#=============================================================


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

    # load vid : )
    vid = imageio.get_reader('/Users/kentchiu/MIT_Vedio/2016-01-21/10.167.10.158_01_20160121082638418_1.mp4')

    #conf = Conf('conf_hub/conf_001.json')
    conf = Conf('conf_hub/conf_001.json')

    # loading model
    clf = joblib.load(conf['model_ph'])

    # initialize feature container
    hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
    cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
    # initialize the object detector
    od = ObjectDetector(clf, hog)




img = vid.get_data(100)[300:440,580:705]


def template_match(refImg, newImg):
    res = cv2.matchTemplate(newImg, refImg, cv2.TM_CCOEFF_NORMED)
    #cv2.TM_CCOEFF_NORMED
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    h , w, _ = refImg.shape
    bottom_right = (top_left[0]+w,top_left[1]+h)
    print (top_left, bottom_right)
    cv2.rectangle(newImg, top_left, bottom_right, 255, 2)
    show(newImg)
    return newImg[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w], (top_left, bottom_right)


cv2.TM_CCOEFF_NORMED
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)


def template_match_v2(refImg, newImg, thresHold):
    res = cv2.matchTemplate(newImg, refImg, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print min_val, max_val, min_loc, max_loc
    top_left = max_loc
    h , w, _ = refImg.shape
    bottom_right = (top_left[0]+w,top_left[1]+h)
    print (top_left, bottom_right)
    cv2.rectangle(newImg, top_left, bottom_right, 255, 2)
    show(newImg)
    return newImg[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w], (top_left, bottom_right)



####################################
# A Good Template is very PowerFul #
####################################


#aa = []
#def func(s):
#    if s ==0:
#        print 'Zero'
#        return 
#    if s%2==0:
#        print 'even'
#        s = s-1
#        aa.append(s)
#        func(s)
#    elif s%2==1:
#        print 'Odd'
#        s = s-1
#        aa.append(s**2)
#        func(s)

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
############
# testing  #
############

s = [3,4,5,6,73,8]
a = [44,33]
for i in s:
    a.append(i)
    a = sequence_container(a,3)
    print a


###########################################################################
# 1. HOG+SVM or Haar or F-R-CNN for object detection (High-critical)      #
# 2. Use Template for Flow-tracking hdfghjk                               #
# 3. If the object is going to leave the camera => re-Recognition         #
# 4. 


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
            hand.append([(x+50+x+50+w)/2,(y+50+h+y+50)/2])
    #show(ref)
            print ((x+50,y+50),(x+50+w,y+50+h))
            hand_roi.append([x+50,y+50,x+50+w,y+50+h])
            # startX, startY, endX, endY
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
def checkState_1(memo):
    '''memo = memoState_1'''
    if len(memo) < 5 :
        return 1
    if memo[-1]==memo[-4] and memo[-4]==memo[-8] and memo[-8]==memo[-12] and memo[-12]== memo[-15]:
        return 0

#---> in a batch way
#---> conf.

def main_script(newImg,stateID):
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
        believe = checkState_1(memoState_1)
        if believe==1:
            print (stateID)
            #-> Output
            global refImg
            refImg = img[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2],:]
            result_img.append(img)
            result_position.append(tarBox)
            #-> Memo
            memoState_1.append(tarBox)
            memoState_1 = sequence_container(memoState_1,15)
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

for i in range(len(vid)):
    main_script(vid.get_data(i),state)

#--------------> Saving Vedio 
output = 'motion_tracking_1.mp4'
height, width, channels = vid.get_data(0).shape

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 5.0, (width, height))

for image in result_img:
    out.write(image) # Write out frame to video

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()




#########
#-------------->TEST
#########

def checkState_1(memo):
    '''memo = memoState_1'''
    if len(memo) < 15 :
        return 1
    if memo[-1]==memo[-4] and memo[-4]==memo[-8] and memo[-8]==memo[-12] and memo[-12]== memo[-15]:
        return 0

def main_script(newImg,stateID):
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
        believe = checkState_1(memoState_1)
        if believe==1:
            print (stateID)
            #-> Output
            global refImg
            refImg = img[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2],:]
            result_img.append(img)
            result_position.append(tarBox)
            #-> Memo
            memoState_1.append(tarBox)
            memoState_1 = sequence_container(memoState_1,15)
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

for i in range(600):
    main_script(vid.get_data(i),state)

#--------------> Saving Vedio 
output = 'motion_tracking_6.mp4'
height, width, channels = vid.get_data(0).shape

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 10.0, (width, height))

for image in result_img:
    out.write(image) # Write out frame to video

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()





#########
#-------------->TEST
#########
def checkState_1(memo):
    '''memo = memoState_1'''
    if len(memo) < 8 :
        return 1
    if memo[-1]==memo[-4] and memo[-4]==memo[-8]:
        return 0

def main_script(newImg,stateID):
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
        # propose of intereted region
        startX,startY,endX,endY = result_position[-1]
        roi = newImg.copy()[(startY-100):(endY-100),(startX+100):(endX+100),:]
        show(roi)
        # template match
        _, tarBox = template_match(refImg, roi)
        # re-correct the tarBox
        a,b,c,d = tarBox
        tarBox = (a+startX,b+startY,c+startX,d+startY)
        print '[*]'
        print tarBox
        #
        cv2.rectangle(newImg, (a+startX,b+startY), (c+startX,d+startY), 255, 2)
        img = newImg
        #
        believe = checkState_1(memoState_1)
        if believe==1:
            print (stateID)
            #-> Output
            global refImg
            refImg = img[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2],:]
            result_img.append(img)
            result_position.append(tarBox)
            #-> Memo
            memoState_1.append(tarBox)
            memoState_1 = sequence_container(memoState_1,8)
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

for i in range(600):
    main_script(vid.get_data(i),state)

#--------------> Saving Vedio 
output = 'motion_tracking_7.mp4'
height, width, channels = vid.get_data(0).shape

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 10.0, (width, height))

for image in result_img:
    out.write(image) # Write out frame to video

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()





#########
#-------------->TEST
#########
def checkState_1(memo):
    '''memo = memoState_1'''
    if len(memo) < 5 :
        return 1
    if memo[-1]==memo[-3] and memo[-3]==memo[-5]:
        return 0

def main_script(newImg,stateID):
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
        # propose of intereted region
        startX,startY,endX,endY = result_position[-1]
        roi = newImg.copy()[(startY-200):(endY-200),(startX+200):(endX+200),:]
        show(roi)
        # template match
        _, tarBox = template_match(refImg, roi)
        # re-correct the tarBox
        a,b,c,d = tarBox
        tarBox = (a+startX,b+startY,c+startX,d+startY)
        print '[*]'
        print tarBox
        #
        cv2.rectangle(newImg, (a+startX,b+startY), (c+startX,d+startY), 255, 2)
        img = newImg
        #
        believe = checkState_1(memoState_1)
        if believe==1:
            print (stateID)
            #-> Output
            global refImg
            refImg = img[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2],:]
            result_img.append(img)
            result_position.append(tarBox)
            #-> Memo
            memoState_1.append(tarBox)
            memoState_1 = sequence_container(memoState_1,5)
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

for i in range(600):
    main_script(vid.get_data(i),state)

#--------------> Saving Vedio 
output = 'motion_tracking_8.mp4'
height, width, channels = vid.get_data(0).shape

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 10.0, (width, height))

for image in result_img:
    out.write(image) # Write out frame to video

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

def video_saving(fileName, fps, imgSequence):
    height, width, channels = imgSequence[0].shape    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(fileName, fourcc, fps, (width, height))    
    for image in imgSequence:
        out.write(image) # Write out frame to video    
    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()




#########
#-------------->TEST XX
#########
def checkState_1(memo):
    '''memo = memoState_1'''
    if len(memo) < 7 :
        return 1
    if memo[-1]==memo[-3] and memo[-3]==memo[-7]:
        return 0

def main_script(newImg,stateID):
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
        ## propose of intereted region
        #startX,startY,endX,endY = result_position[-1]
        #roi = newImg.copy()[(startY-200):(endY-200),(startX+200):(endX+200),:]
        #show(roi)
        ## template match
        #_, tarBox = template_match(refImg, roi)
        ## re-correct the tarBox
        #a,b,c,d = tarBox
        #tarBox = (a+startX,b+startY,c+startX,d+startY)
        #print '[*]'
        #print tarBox
        ##
        #cv2.rectangle(newImg, (a+startX,b+startY), (c+startX,d+startY), 255, 2)
        #img = newImg
        ##
        img, tarBox = template_match(refImg, newImg)
        believe = checkState_1(memoState_1)
        if believe==1:
            print (stateID)
            #-> Output
            global refImg
            refImg = img[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2],:]
            result_img.append(img)
            result_position.append(tarBox)
            #-> Memo
            memoState_1.append(tarBox)
            memoState_1 = sequence_container(memoState_1,7)
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

for i in range(600):
    main_script(vid.get_data(i),state)

#--------------> Saving Vedio 
video_saving('motion_tracking_8.mp4',10.0,result_img)

#def video_saving(fileName, fps, imgSequence):
#    height, width, channels = imgSequence[0].shape    
#    # Define the codec and create VideoWriter object
#    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
#    out = cv2.VideoWriter(fileName, fourcc, fps, (width, height))    
#    for image in imgSequence:
#        out.write(image) # Write out frame to video    
#    # Release everything if job is finished
#    out.release()
#    cv2.destroyAllWindows()


#=========================================
def checkState_1(memo):
    '''memo = memoState_1'''
    if len(memo) < 15 :
        return 1
    if memo[-1]==memo[-3] and memo[-3]==memo[-15]:
        return 0

def main_script(newImg,stateID):
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
            memoState_0 = sequence_container(memoState_0,15)
            return
    else:
        ###############################################
        #img, tarBox = template_match(refImg, newImg) #
        #believe = checkState_1(memoState_1)          #
        ###############################################
        # propose of intereted region
        startX,startY,endX,endY = result_position[-1]
        def upZero(a):
            if a < 0:
                return 0
            else :
                return a
        roi = newImg.copy()[upZero(startY-200):upZero(endY-200),(startX+200):(endX+200),:]
        show(roi)
        # template match
        _, tarBox = template_match(refImg, roi)
        # re-correct the tarBox
        a,b,c,d = tarBox
        tarBox = (a+startX,b+startY,c+startX,d+startY)
        print '[*]'
        print tarBox
        #
        cv2.rectangle(newImg, (a+startX,b+startY), (c+startX,d+startY), 255, 2)
        img = newImg
        believe = checkState_1(memoState_1)  
        #
        if believe==1:
            print (stateID)
            #-> Output
            global refImg
            refImg = img[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2],:]
            result_img.append(img)
            result_position.append(tarBox)
            #-> Memo
            memoState_1.append(tarBox)
            memoState_1 = sequence_container(memoState_1,15)
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

for i in range(600):
    main_script(vid.get_data(i),state)

#--------------> Saving Vedio 
video_saving('motion_tracking_10.mp4',10.0,result_img)





#=========================================
def checkState_1(memo):
    '''memo = memoState_1'''
    if len(memo) < 30 :
        return 1
    if memo[-1]==memo[-3] and memo[-10]==memo[-30]:
        return 0

def main_script(newImg,stateID):
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
            memoState_0 = sequence_container(memoState_0,15)
            return
    else:
        ###############################################
        #img, tarBox = template_match(refImg, newImg) #
        #believe = checkState_1(memoState_1)          #
        ###############################################
        # propose of intereted region
        startX,startY,endX,endY = result_position[-1]
        def upZero(a):
            if a < 0:
                return 0
            else :
                return a
        roi = newImg.copy()[upZero(startY-200):upZero(endY-200),(startX+200):(endX+200),:]
        show(roi)
        # template match
        refImg = cv2.Canny(refImg,10,200)
        roi = cv2.Canny(roi, 10, 200)

        _, tarBox = template_match_2(refImg, roi)
        # re-correct the tarBox
        a,b,c,d = tarBox
        tarBox = (a+startX,b+startY,c+startX,d+startY)
        print '[*]'
        print tarBox
        #
        cv2.rectangle(newImg, (a+startX,b+startY), (c+startX,d+startY), 255, 2)
        img = newImg
        believe = checkState_1(memoState_1)  
        #
        if believe==1:
            print (stateID)
            #-> Output
            global refImg
            refImg = img[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2],:]
            result_img.append(img)
            result_position.append(tarBox)
            #-> Memo
            memoState_1.append(tarBox)
            memoState_1 = sequence_container(memoState_1,30)
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

for i in range(600):
    main_script(vid.get_data(i),state)

#--------------> Saving Vedio 
# Canny Match
video_saving('motion_tracking_11.mp4',10.0,result_img)#15

video_saving('motion_tracking_11_f30.mp4',20.0,result_img) # frame correlation = 30 

#=========================|
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

#template = cv2.Canny(refImg, 10, 200)


#--------------> Test CamShift
import numpy as np
import cv2

cap = cv2.VideoCapture('slow.flv')
cap = cv2.VideoCapture('/Users/kentchiu/MIT_Vedio/2016-01-21/10.167.10.158_01_20160121082638418_1.mp4')


# take first frame of the video
ret,frame = cap.read()

frame = vid.get_data(1)
# setup initial location of window
img, tarBox = cascade_test_15_c3(frame)
r,h,c,w = 250,90,400,125  # simply hardcoded the values
img, tarBox = cascade_test_15_c3(frame)
c,r,w,h = tarBox[0],tarBox[1],tarBox[2]-tarBox[0], tarBox[3]-tarBox[1]
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
show(roi)
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )


frame = vid.get_data(2)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
# apply meanshift to get the new location
ret1, track_window = cv2.CamShift(dst, track_window, term_crit)
ret2, track_window2 = cv2.meanShift(dst, track_window, term_crit)
x,y,w,h = track_window2
img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
show(img2)

#--> ret : central 
#--> track_window = (c,r,w,h)
c = int(max(ret[0]))
r = int(max(ret[1]))
cv2.rectangle(frame, (c,r), (c+w,r+h), 255, 2)
show(frame)

# Draw it on image
pts = cv2.boxPoints(ret)
pts = np.int0(pts)
img2 = cv2.polylines(frame,[pts],True, 255,2)
show(img2)

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
        show(img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)

    else:
        break

cv2.destroyAllWindows()
cap.release()







###################
#----------> In a real-time way
memo = []
memoState_0 = []
memoState_1 = []
result_img = []
result_position =[]
stateID = 0
while(1):
    _, frame = cap.read()
    memo.append(frame)
    memo = sequence_container(a,3)
    #--->init
    main_script(frame,stateID)

frame_id = []
hand_temp = []

for i in range(630):
    # init
    img = vid.get_data(i*3)
    frame_id.append(i*3)
    hand =cascade_test_15_c3(img)
    hand_temp.append(hand)


#------------------------



#












