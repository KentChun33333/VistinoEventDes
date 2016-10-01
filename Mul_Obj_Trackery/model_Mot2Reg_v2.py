__Author__='Kent Chiu'
__CreateDate__ = '2016-08-08'

# This Script is based on the previous version from Motion_Recognition
# For realtime process, could ref it

import numpy as np
import argparse
import imutils
import cv2
import motion_interpretation
from scipy.stats.mstats import mode as statics_mode

from HUB_dictionary.motion_dictionary  import Motion_Dictionary
from HUB_dictionary.path_dictionary    import Path_DTW_Dictionary
from HUB_dictionary.gesture_dictionary import Gesture_DTW_Dictionary
from HUB_Model.multi_recog import HaarCV_Recognizor, PureScrewDriverRecog

from com_func.conf import Conf




class Recog2Track():
    def __init__(self, mulRecog, modelNameList, path_recognition_Flag=True):
        ''' mulRecog = [M1, M2, ...]
            modelNameList = [str(M1), str(M2)...] '''
        self.numFrames = 0
        self.modelNameList = modelNameList
        self.modelNum = len(mulRecog)
        ##########################################

        # this is the collection that contains the obj center-point
        self.position = []

        # this is the motion- number representation with out any cut, trim,
        # this variable is used to output the training data or ...etc
        self.motionALL = [] # OutPutModel...

        # this variable is the same as above, just trimmed
        self.motionSeq = []

        # max likelihood for the most possible motion
        self.motionLikehoodSeq = []

        # this variable is contain the ref_img that for fast match
        self.refImgforMatch = []

        # this variable is contrain the Flag of state
        self.trackingFlag = []

        # According to how many obj , we init_corresponding variables
        #
        for i in range(self.modelNum):
            # Below structure is [ [] ,  []]
            # or [ 1,0  ] for Flag
            self.position.append([])
            self.motionALL.append([])
            self.motionSeq.append([])
            self.motionLikehoodSeq.append([])
            self.refImgforMatch.append([])
            self.trackingFlag.append(0)
        ###########################################
        self.path_recognition = path_recognition_Flag
        self.pathRegStr = "None"
        self.actionStr = "None"
        self.recogModels = mulRecog


    def tracking_mode(self,):
        pass

    def template_match(self, newImg,refImg , thresHold=0.85):
        '''
        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where( res >= threshold)
        -----------------------------------
        Output = NweImg and TarBox
        '''
        res = cv2.matchTemplate(newImg, refImg, cv2.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # if the match is quite  not like the original
        if max_val < thresHold:
            return newImg, []

        # startX , startY
        top_left = max_loc
        h , w, _ = refImg.shape
        bottom_right = (top_left[0]+w,top_left[1]+h)

        cv2.rectangle(newImg, top_left, bottom_right, 255, 2)
        startX , startY = top_left[0], top_left[1]
        endX, endY = bottom_right
        tarBox = (startX, startY, endX, endY)
        print tarBox
        return newImg, tarBox


    def motion_feature_extraction(self, imgAFrecog ,pointsSet,modelID):
        '''
        imgAFrecog : the raw img after static recognition
        pointsSet  : the location of the objects
        modelID    : the name that we predifined when iniciation the class
        '''
        Dictionary = Motion_Dictionary()
        PathDictionary = Path_DTW_Dictionary()
        if len(pointsSet)>3 :
            last_point = pointsSet[-2]
            tmp_motion = motion_interpretation.interpreter(last_point,pointsSet[-1])
            #
            self.motionALL[modelID].append(tmp_motion) # this is all motion
            self.motionSeq[modelID].append(tmp_motion) # this is time-segment motion
            # Compute the max_Likelihood
            self.motionSeq[modelID]      = self.seq_clip(self.motionSeq[modelID],10)
            motionLikehood = statics_mode(self.motionSeq[modelID])[0]
            self.motionLikehoodSeq[modelID].append(motionLikehood)
            motionStr      = Dictionary.check(motionLikehood)
            # %.2f = float with 2 dicimal
            modelName = self.modelNameList[modelID]
            cv2.putText(imgAFrecog,
                "The %s is %s" % (modelName,motionStr),
                 (20, 50*(1+modelID)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 6)
            if len(self.motionLikehoodSeq[modelID])>10 and (self.path_recognition is True):

                # put it in small LSTM or RNN for NPL
                # put it into DTW search/match
                PathRegStr_tmp = PathDictionary.search(self.motionLikehoodSeq[modelID][-7:])
                if PathRegStr_tmp == 'None':
                    pass
                else:
                    cv2.putText(imgAFrecog,
                        "The %s is %s" % (modelName, PathRegStr_tmp),
                        (20, 100*(1+modelID)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 6)
                    self.pathRegStr = PathRegStr_tmp

            # This part spared by not doing path Recognition,
            # if we want to do path recog, please ref 2016MIT/Motion_Recognition
        return imgAFrecog

    def have_refimg(self, imgInput,tarBox,channels ):
        if channels==3:
            # process RBG, HSC, data (3 channels)
            return imgInput[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2],:]
        else :
            # process depth data or gray-scale data
            return imgInput[tarBox[1]:tarBox[3],tarBox[0]:tarBox[2]]

    def seq_clip(self, inPut,seqLen):
        if len(inPut)<seqLen:
            return inPut
        return inPut[-seqLen:]

    def perform_VID_analysis(self, startFrame, endFrame, vid):
        '''vidObj based on the image io OBJ'''
        self.numFrames = startFrame
        imgSeq = []
        assert endFrame > startFrame and endFrame < vid.get_length()
        while self.numFrames <= endFrame:
            img = vid.get_data(self.numFrames)
            NewImg = img.copy()

            # this operation is to create a black board for print the text
            cv2.rectangle(NewImg,(0,0),(800,125),(10,10,10),-1)

            for modelID, model in enumerate(self.recogModels):
                ############################
                # add the criteria of Flag #
                ############################
                if self.trackingFlag[modelID]==0:
                    NewImg, tarBox = model.detect(NewImg)

                    ################################################################################
                    print 'recognitions mode'
                    print 'Processing : ( Model : {}, Frame : {} )'.format(modelID, self.numFrames)
                    if len(tarBox)>0: # obj been detected
                        assert len(tarBox)==4
                        position = ((tarBox[0]+tarBox[2])/2, (tarBox[0]+tarBox[2])/2)
                        self.position[modelID].append(position)
                        ################################
                        # output the refImg by objects #
                        ################################
                        self.refImgforMatch[modelID] = self.have_refimg(NewImg,tarBox,3)

                        #####################################
                        # if detected, than change the Flag #
                        #####################################
                        self.trackingFlag[modelID]=1

                    else: # Treat as Stationary
                        if len(self.position[modelID])>0:
                            self.position[modelID].append(self.position[modelID][-1])
                    NewImg = self.motion_feature_extraction(NewImg,self.position[modelID], modelID)

                    #

                elif self.trackingFlag[modelID]==1:
                    NewImg, tarBox = self.template_match(NewImg, self.refImgforMatch[modelID] )

                    ################################################################################
                    print ('tracking_mode')
                    print 'Processing : ( Model : {}, Frame : {} )'.format(modelID, self.numFrames)
                    if len(tarBox)>0: # obj been detected
                        assert len(tarBox)==4
                        position = ((tarBox[0]+tarBox[2])/2, (tarBox[0]+tarBox[2])/2)
                        self.position[modelID].append(position)
                        ################################
                        # output the refImg by objects #
                        ################################
                        self.refImgforMatch[modelID] = self.have_refimg(NewImg,tarBox,3)

                        #####################################
                        # if detected, than change the Flag #
                        #####################################
                        self.trackingFlag[modelID]=1

                    else: # Treat as Stationary
                        if len(self.position[modelID])>0:
                            self.position[modelID].append(self.position[modelID][-1])
                        #####################################
                        # if detected, than change the Flag #
                        #####################################
                        self.trackingFlag[modelID]=0
                    NewImg = self.motion_feature_extraction(NewImg,self.position[modelID], modelID)
                    ################################################################################

                self.numFrames+=1
                imgSeq.append(NewImg)
        return imgSeq

    def perform_WebCam_analysis(self, capIMG):
        '''vidObj based on the image io OBJ'''
        NewImg = capIMG.copy()
        # for out put on the motino recognition
        cv2.rectangle(NewImg,(0,0),(800,125),(10,10,10),-1)
        for modelID, model in enumerate(self.recogModels):
            NewImg, tarBox = model.detect(NewImg)
            print 'Processing : ( Model : {}, Frame : {} )'.format(modelID, self.numFrames)
            if len(tarBox)>0: # obj been detected
                assert len(tarBox)==4
                position = ((tarBox[0]+tarBox[2])/2, (tarBox[0]+tarBox[2])/2)
                self.position[modelID].append(position)
            else: # Treat as Stationary
                if len(self.position[modelID])>0:
                    self.position[modelID].append(self.position[modelID][-1])
            NewImg = self.motion_feature_extraction(NewImg,self.position[modelID], modelID)
        self.numFrames+=1
        self.pathRegStr = 'None'
        return NewImg

    def reinitialising_parameter(self):
        self.position = []
        self.motionALL = [] # OutPutModel...
        self.motionSeq = []
        self.motionLikehoodSeq = []
        self.refImgforMatch = []
        for i in range(self.modelNum):
            self.position.append([])
            self.motionALL.append([])
            self.motionSeq.append([])
            self.motionLikehoodSeq.append([])
            self.refImgforMatch.append([])
        self.pathRegStr = "None"
        self.actionStr = "None"

    def feature_describe(self):
        return self.motionLikehoodSeq

class Dispach_Agent_Static_to_Dynamic():
    def __init__():
        pass

def template_match_center_point(refImg, newImg, thresHold=0.88):
    '''
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    '''
    res = cv2.matchTemplate(newImg, refImg, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # if the match is quite  not like the original
    if max_val < thresHold:
        return None
    print min_val, max_val, min_loc, max_loc
    top_left = max_loc
    h , w, _ = refImg.shape
    bottom_right = (top_left[0]+w,top_left[1]+h)
    print (top_left, bottom_right)
    cv2.rectangle(newImg, top_left, bottom_right, 255, 2)
    #show(newImg)
    #NewTemplate =newImg[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
    cx = int(top_left[0]+0.5*w)
    cy = int(top_left[1]+0.5*h)
    return cx, cy

if __name__ == '__main__':
    import imageio
    vid=imageio.get_reader('~/MIT_Vedio/2016-01-21/10.167.10.158_01_20160121082638418_1.mp4')
    model_1 = HaarCV_Recognizor()
    model_2 = PureScrewDriverRecog(Conf('conf_hub/conf_pureScrewDriver_2.json'))
    ReTesT = Recog2Track([model_1,model_2],['Hand', 'SkrewDriver'])
    output = ReTesT.perform_VID_analysis(270,350,vid)
'''
@ MAC OS
model_1 = HaarCV_Recognizor()
model_2 = PureScrewDriverRecog(Conf('conf_hub/conf_pureScrewDriver_2.json'))

@ ROG computer
model_1 = HaarCV_Recognizor(
    'C:/Users/kentc/Documents/GitHub/VistinoEventDes/2016_MIT/Auto_HOG_SVM/model_hub/svm/PureScrewDriver_2.model')
PureScrewDriverRecog(Conf('C:/Users/kentc/Documents/GitHub/VistinoEventDes/2016_MIT/Auto_HOG_SVM/conf_hub/conf_pureScrewDriver_2.json'))

ReTesT = Recog2Track([model_1,model_2],['Hand', 'SkrewDriver'])
output = ReTesT.perform_VID_analysis(270,350,vid)
video_saving('~/MIT_Vedio/test_0812_New3.mp4',8.0,output)
'''

#Model_2 = PureScrewDriverRecog(Conf('C:/Users/kentc/Documents/GitHub/VistinoEventDes/2016_MIT/Auto_HOG_SVM/conf_hub/conf_pureScrewDriver_2.json'))
#model = Multi_Model_Iterative_Detect([Model_1,Model_2])#

#model.showDetect(vid.get_data(12))#

#show(Model_2.detect(vid.get_data(300), pro = 0.6)[0])
#show(Model_2.detect(vid.get_data(300), pro = 0.8)[0])
#show(Model_2.detect(vid.get_data(300), pro = 0.85)[0])
#model = Multi_Model_Iterative_Detect([Model_1,Model_2])
#model.showDetect(vid.get_data(305))
#model.showDetect(vid.get_data(315))
#model.showDetect(vid.get_data(1))
#model.showDetect(vid.get_data(12))

# vid = imageio.get_reader('D:/2016-01-21//10.167.10.158_01_20160121082638418_1.mp4')

'''
PureScrewDriverRecog(Conf('C:/Users/kentc/Documents/GitHub/VistinoEventDes/2016_MIT/Auto_HOG_SVM/conf_hub/conf_pureScrewDriver_2.json'))
model.showDetect(vid.get_data(305))

model_1 = HaarCV_Recognizor('model_hub/opencv_cascade/frontalFace10/haarcascade_frontalface_alt.xml')
'''
