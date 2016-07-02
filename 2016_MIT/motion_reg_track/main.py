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
    conf = Conf('conf_hub/conf_depth_L1_finger_4.json')

    # loading model
    clf = joblib.load(conf['model_ph'])

    # initialize feature container
    hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
    cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
    # initialize the object detector
    od = ObjectDetector(clf, hog)


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

def neighbor_attention(imgInput,tarBox, attenScale=0.2):
    '''
    attenScale : would depend on the tarbox, default = 0.35
    output : propose of ROI and new_tarbox
    '''
    startX, startY, endX, endY  = tarBox
    width = endX-startX
    height = endY-startY
    startX = int(max(startX-attenScale*width,0))
    startY = int(max(startY-attenScale*height,0))
    maxY, maxX = imgInput.shape[0], imgInput.shape[1]
    endX = int(min(endX+attenScale*width,maxX))
    endY = int(min(endY+attenScale*height, maxY))
    tarBox = [startX, startY, endX, endY]
    roi = imgInput[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2]]
    return roi, tarBox




def crop_tarBox(imgInput, tarBox):
    return imgInput[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2]]

def have_refimg(imgInput,tarBox,channels ):
    if channels==3:
        # process RBG, HSC, data (3 channels)
        return imgInput[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2],:]
    else :
        # process depth data or gray-scale data
        return imgInput[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2]]

###########################################################################
# 1. HOG+SVM or Haar or F-R-CNN for object detection (High-critical)      #
# 2. Use Template for Flow-tracking hdfghjk                               #
# 3. If the object is going to leave the camera => re-Recognition         #
# 4. 


#-------------------------------------------------------------------------
def cascade_test_15_c3(imgInput,proThred=15):
    hand_5 = cv2.CascadeClassifier(
    '/Users/kentchiu/MIT_Vedio/Rhand_no_tools/cascade.xml')
    ref = imgInput.copy()
    hand5 = hand_5.detectMultiScale(
        ref.copy()[50:550,50:800],
        scaleFactor=1.2,
        minNeighbors=proThred, #35 ==> 1
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

def template_match_gray(refImg, newImg):
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
    if len(memo) < 15 :
        return 1
    if memo[-1]==memo[-4] and memo[-4]==memo[-8] \
    and memo[-8]==memo[-12] and memo[-12]== memo[-15]:
        return 0
#---> in a batch way
#---> conf.

def test_with_pro_depth_size(rawImg,pro):
    ''' Use this fuc when ### above are first implement'''
    #img = vid.get_data(frame_id)
    roi = auto_resized(rawImg,conf['train_resolution'])
    (boxes, probs) = od.detect(roi, winStep=conf["step_size"], winDim=conf["sliding_size"],
    pyramidScale=1.1, minProb=pro)
    # since for size effect with the camera, pyramidScale = 1.001, mnust>1, 
    # if positive size would change, we have to use 1.5 or 2 ...etc 
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    orig = roi.copy()
    # loop over the allowed bounding boxes and draw them
    hand_roi=[]
    for (startX, startY, endX, endY) in pick:
        if endX-startX>30:
            pass
        else:
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
            print (startX, startY), (endX, endY)
            hand_roi.append([startX, startY, endX, endY])
    
    #show(orig)

    ###############
    if len(hand_roi)>1:
        result_box = max(hand_roi)
    elif len(hand_roi)==1:
        result_box = hand_roi.pop()
    else:
        result_box = 0
    #####
    return orig, result_box

def have_refimg(imgInput,tarBox,channels ):
    if channels==3:
        # process RBG, HSC, data (3 channels)
        return imgInput[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2],:]
    else :
        # process depth data or gray-scale data
        return imgInput[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2]]


def main_script(newImg,stateID):
    # Add this line in order to resize the image to keep the output image 
    # all at the same size
    newImg = auto_resized(newImg,conf['train_resolution'])
    #-----init-----
    global memoState_1
    if stateID==0:
        #img, tarBox = cascade_test_15_c3(newImg)
        img, tarBox = test_with_pro_depth_size(newImg,0.87)
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
            refImg = have_refimg(img, tarBox,channels=2)
            #-> Clean memo_0
            memoState_0=[]
            return 
        else:
            memoState_0.append(img)
            memoState_0 = sequence_container(memoState_0,5)
            return
    else:
        global refImg
        global attention
        
        if attention == 0 : 
            img, tarBox = template_match(refImg, newImg)
        else:
            #[*] add below to match only the neighbor hood
            print '[*] Attention_mode'
            att_img , att_tarBox = neighbor_attention(newImg,result_position[-1])
            shiftX = att_tarBox[0]
            shiftY = att_tarBox[1]
            print '[#]'+str(att_tarBox)

            _ , tarBox = template_match_gray(refImg, crop_tarBox(newImg,att_tarBox))
            startX, startY , endX, endY = tarBox

            tarBox = [startX+shiftX, startY+shiftY , endX+shiftX, endY+shiftY ]
            startX, startY , endX, endY = tarBox

            print '[!] new Tarbox'+str(tarBox)
            img = newImg.copy()
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

        believe = checkState_1(memoState_1)
        if believe==1:
            print (stateID)
            #-> Output
            global refImg
            refImg = have_refimg(img, tarBox,channels=2)
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

#-----------------------------------
# Give Depth Target Folder 
# this motion_tracking with switching is different from single detector
list_dir = os.listdir('/Users/kentchiu/MIT_Vedio/3D_DataSet/L_1_finger/')[1:]
for i in range(55,1000):
    main_script(imread('/Users/kentchiu/MIT_Vedio/3D_DataSet/L_1_finger/'+list_dir[i]),state)

dep_video_saving('conf_depth_L1_finger_4_MOT.mp4',10.0,result_img)



######################################################################
#
# change check-state1 to always 1
def checkState_1(memo):
    '''memo = memoState_1'''
    return 1

memo = []
memoState_0 = []
memoState_1 = []
result_img = []
result_position =[]
attention = 0 # open_attention_mode
state = 0 # variable better not the same as one in the function

list_dir = os.listdir('/Users/kentchiu/MIT_Vedio/3D_DataSet/L_1_finger/')[1:]
for i in range(60,1000):
    main_script(imread('/Users/kentchiu/MIT_Vedio/3D_DataSet/L_1_finger/'+list_dir[i]),state)

dep_video_saving('conf_depth_L1_finger_4_MOT_B.mp4',10.0,result_img)
#
#
######################################################################


######################################################################
#
# change check-state1 to always 1
def checkState_1(memo):
    '''memo = memoState_1'''
    return 1

memo = []
memoState_0 = []
memoState_1 = []
result_img = []
result_position =[]
attention = 1 # open_attention_mode
state = 0 # variable better not the same as one in the function

list_dir = os.listdir('/Users/kentchiu/MIT_Vedio/3D_DataSet/L_1_finger/')[1:]
for i in range(65,1000):
    main_script(imread('/Users/kentchiu/MIT_Vedio/3D_DataSet/L_1_finger/'+list_dir[i]),state)

dep_video_saving('conf_depth_L1_finger_4_MOT_C.mp4',10.0,result_img)
#
#
######################################################################



memo = []
memoState_0 = []
memoState_1 = []
result_img = []
result_position =[]
attention = 0 # open_attention_mode
state = 0 # variable better not the same as one in the function

class XXXX():
    def __init__(self, memo, memoState_1, memoState_0,
    result_img, result_position, attention, state=0,detector):
        self.memo = memo
        self.memoState_1 = memoState_1
        self.memoState_0 = memoState_0
        self.result_img = result_img
        self.result_position =result_position
        self.attention = attention
        self.state = state 
        # detector is a model(function as object)
        self.detector = detector
        self.refImg 

    def crop_tarBox(self, imgInput, tarBox):
        return imgInput[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2]]

    def have_refimg(self,imgInput,tarBox,channels ):
        if channels==3:
            # process RBG, HSC, data (3 channels)
            return imgInput[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2],:]
        else :
            # process depth data or gray-scale data
            return imgInput[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2]]

    def neighbor_attention(self,imgInput,tarBox, attenScale=0.2):
        '''
        attenScale : would depend on the tarbox, default = 0.2
        output : propose of ROI and new_tarbox
        '''
        startX, startY, endX, endY  = tarBox
        width = endX-startX
        height = endY-startY
        startX = int(max(startX-attenScale*width,0))
        startY = int(max(startY-attenScale*height,0))
        maxY, maxX = imgInput.shape[0], imgInput.shape[1]
        endX = int(min(endX+attenScale*width,maxX))
        endY = int(min(endY+attenScale*height, maxY))
        tarBox = [startX, startY, endX, endY]
        roi = imgInput[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2]]
        return roi, tarBox

    def checkState_1(self,memo):
        '''memo = memoState_1'''
        return 1

    @staticmethod
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

    def checkState_0(self):
        return 1

    def main_script(self,newImg):
        # Add this line in order to resize the image to keep the output image 
        # all at the same size
        newImg = auto_resized(newImg,conf['train_resolution'])
        #-----init-----
        if self.state==0:
            #img, tarBox = cascade_test_15_c3(newImg)
            img, tarBox = self.detector(newImg,0.87)
            if tarBox==0:
                print '[*] the tarBox is None as recognition'
                self.result_img.append(img)
                self.result_position.append(tarBox)
                return
            believe = checkState_0()
            if believe==1:
                #-> Output 
                self.result_img.append(img)
                self.result_position.append(tarBox)
                #-> Init extra-variable with state 1
                self.state=1 
                self.refImg = have_refimg(img, tarBox,channels=2)
                #-> Clean memo_0
                self.memoState_0=[]
                return 
            else:
                self.memoState_0.append(img)
                self.memoState_0 = sequence_container(self.memoState_0,5)
                return
        else:
            
            if self.attention == 0 : 
                img, tarBox = template_match(refImg, newImg)
            else:
                #[*] add below to match only the neighbor hood
                print '[*] Attention_mode'
                att_img , att_tarBox = neighbor_attention(newImg,self.result_position[-1])
                shiftX = att_tarBox[0]
                shiftY = att_tarBox[1]

                _ , tarBox = template_match_gray(refImg, crop_tarBox(newImg,att_tarBox))
                startX, startY , endX, endY = tarBox    

                tarBox = [startX+shiftX, startY+shiftY , endX+shiftX, endY+shiftY ]
                startX, startY , endX, endY = tarBox    

                print '[!] new Tarbox'+str(tarBox)
                img = newImg.copy()
                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)    

            believe = checkState_1(self.memoState_1)
            if believe==1:
                #-> Output
                self.refImg = have_refimg(img, tarBox,channels=2)
                self.result_img.append(img)
                self.result_position.append(tarBox)
                #-> Memo
                self.memoState_1.append(tarBox)
                self.memoState_1 = sequence_container(self.memoState_1,15)
                return
            else:
                self.state=0
                #-> Clean memo
                self.memoState_1=[]
                return


