
import multiprocess
import logging
import gc
import unittest





import motion_interpretation
from HUB_dictionary.path_dictionary    import Path_DTW_Dictionary

import common_func
import gc
# http://stackoverflow.com/questions/22440421/python-is-the-garbage-collector-run-before-a-memoryerror-is-raised
# if there are hidden reference cycles
# use gc.collect() to manually release the memory in each iteration


def singleton(class_):
  # when the decorator be excuted
  # this variable would be encapsure in this decorater
  instances = {}
  def getinstance(*args, **kwargs):
    if class_ not in instances:
        instances[class_] = class_(*args, **kwargs)
    return instances[class_]
  return getinstance

@singleton
class VidMemo(object):
    def __init__(self):
        self.boxes={}
        self.prob ={}

    def addNode(self, labelID, box, prob):
        '''lablerID could be label or objectName'''
        assert type(box)==tuple
        self.boxes[labelID].append(box)
        self.prob[labelID].append(prob)

    def extratResult(self,labelID):
        try :
            return self.boxes[labelID], self.prob[labelID]
        except Exception as err:
            print (err)


class DTWAnalyzer2D(DynamicModel):
    def __init__(self, objName):
        self.model = Path_DTW_Dictionary()
        self.objName = objName
        self.positionSeq = []
        self.motionSeq = []
        self.motInterpret = motion_interpretation.interpreter

    def add_Memo(self, Memo):
        if hasattr(Memo, 'isNode'):
            self.Memo.append(Memo)

    def predict(self, cur_objCenter, cur_objSize=20):
        '''
        objectSize = w, h
        '''
        self.motinoALL.append(self.motInterpret(last_point,pointsSet[-1], min_noiseDistance))
        if len(self.motionALL)<10:
            return 'Under Analysis'
        motionStr = self.model.search(self.motionALL[-10:])
        return motionStr

    def predict_proba(self, seqData, objSize):
        pass


__Author__='Kent Chiu'
__CreateDate__ = '2016-08-08'

# This Script is based on the previous version from Motion_Recognition
# For realtime process, could ref it

import numpy as np
import os
import argparse
import imutils
import cv2
import motion_interpretation
from scipy.stats.mstats import mode as statics_mode

from HUB_dictionary.motion_dictionary  import Motion_Dictionary
from HUB_dictionary.path_dictionary    import Path_DTW_Dictionary
from HUB_dictionary.gesture_dictionary import Gesture_DTW_Dictionary
from HUB_Model.multi_recog import HaarCV_Recognizor, PureScrewDriverRecog
from HUB_Model import StaticModel , Tracker
from com_func.conf import Conf
from com_func.video_io import video_saving_IO,video_saving_CV
import dlib
import gc
import logging


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
        # Tricky adding Variables, looks redundant at first glence


        # max likelihood for the most possible motion
        self.motionLikehoodSeq = []

        # this variable is contain the ref_img that for fast match
        self.refImgforMatch = []

        #
        self.tracker = []

        # this variable is contrain the Flag of state
        self.trackingFlag = []

        # According to how many obj , we init_corresponding variables
        #
        for i in range(self.modelNum):
            # Below structure is [ [] ,  []]
            # or [ 1,0  ] for Flag
            self.position.append([])
            self.motionALL.append([])
            self.tracker.append(Tracker())
            self.motionLikehoodSeq.append([])
            self.refImgforMatch.append([])
            self.trackingFlag.append(0)
        ###########################################
        self.path_recognition = path_recognition_Flag
        self.pathRegStr = "None"
        self.actionStr = "None"
        self.recogModels = mulRecog


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
        return newImg, tarBox

    def motion_feature_extraction(self, imgAFrecog ,pointsSet,modelID, tarBox_w_h):
        '''
        imgAFrecog : the raw img after static recognition
        pointsSet  : the location of the objects
        modelID    : the name that we predifined when iniciation the class
        '''
        w,h = tarBox_w_h

        motionDictionary = Motion_Dictionary()
        PathDictionary = Path_DTW_Dictionary()
        if len(pointsSet)>3 :
            last_point = pointsSet[-2]

            # ==========================================================
            # this noiseDistance Should be the associated to the tarBox
            # Set the min Distance of object to see as Stationary
            # minValue : 25 is relatived to the raw_Frame_Resolusionn
            effectLength=(w**2+h**2)**(.5)
            min_noiseDistance = min(int(0.07*effectLength),25)

            # interpreter the mostlikly motion
            tmp_motion = motion_interpretation.interpreter(last_point,pointsSet[-1], min_noiseDistance)

            self.motionALL[modelID].append(tmp_motion) # this is all motion


            #=============================================================
            # motion likelihood
            # (likelihood is only useful if hand-gesture recognition)
            # if using likelihood, should not see as static

            #motionLikehood = statics_mode(self.motionSeq[modelID])[0]
            #self.motionLikehoodSeq[modelID].append(motionLikehood)
            motionStr = PathDictionary.search(self.motionALL[modelID][-10:])
            # %.2f = float with 2 dicimal
            modelName = self.modelNameList[modelID]
            cv2.putText(imgAFrecog,
                "ID:%s, %s is %s" % (str(self.numFrames), modelName,motionStr),
                 (20, 50*(1+modelID)),
                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 6)
            if len(self.motionLikehoodSeq[modelID])>10 and (self.path_recognition is True):
                pass

                # =====================================
                # put it in small LSTM or RNN for NPL
                # put it into DTW search/match
                #PathRegStr_tmp = PathDictionary.search(self.motionLikehoodSeq[modelID][-7:])
                if PathRegStr_tmp == 'None':
                    pass
                else:
                    pass
                    #cv2.putText(imgAFrecog,
                    #    "The %s is %s" % (modelName, PathRegStr_tmp),
                    #    (20, 100*(1+modelID)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 6)
                    #self.pathRegStr = PathRegStr_tmp

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


    def perform_VID_analysis(self,
                             startFrame,
                             endFrame,
                             vid,
                             frameInterval=1,
                             rollback=45):
        '''vidObj based on the image io OBJ'''
        self.numFrames = startFrame
        imgSeq = []
        assert endFrame > startFrame and endFrame < vid.get_length()
        while self.numFrames <= endFrame:
            img = vid.get_data(self.numFrames)
            NewImg = img.copy()

            # this operation is to create a black board for print the text
            cv2.rectangle(NewImg,(0,0),(800,125),(10,10,10),-1) # Vis Level

            for modelID, model in enumerate(self.recogModels):

                # add the criteria of Flag
                if self.trackingFlag[modelID]==0:
                    NewImg, tarBox = model.detect(NewImg)
                    print 'recognitions mode' # Info Lel
                    print 'Processing : ( Model : {}, Frame : {} )'.format(modelID, self.numFrames)
                    if len(tarBox)>0: # obj been detected
                        assert len(tarBox)==4
                        position = ((tarBox[0]+tarBox[2])/2, (tarBox[0]+tarBox[2])/2)
                        self.position[modelID].append(position)

                        #===============================
                        # init the tracking model
                        # output the refImg by objects

                        # tarBox enlargement
                        w = tarBox[2]-tarBox[0]
                        h = tarBox[3]-tarBox[1]
                        tarBox[0] = tarBox[0]-min(int(0.1*w),15)
                        tarBox[1] = tarBox[1]-min(int(0.1*h),15)
                        tarBox[2] = tarBox[2]+min(int(0.1*w),15)
                        tarBox[3] = tarBox[3]+min(int(0.1*h),15)

                        self.tracker[modelID].start_track(NewImg, tarBox)
                        #self.refImgforMatch[modelID] = self.have_refimg(NewImg,tarBox,3)

                        # if detected, than change the Flag
                        self.trackingFlag[modelID]=1
                        NewImg = self.motion_feature_extraction(NewImg,
                                    self.position[modelID], modelID, (w,h))

                    else: # if cant not recog => break
                        if len(self.position[modelID])>0:
                            pass
                            #self.position[modelID].append(self.position[modelID][-1])

                elif self.trackingFlag[modelID]==1:
                    # =============================
                    # Tracking level
                    # =============================
                    # Attention Match Method Below
                    # NewImg, tarBox = self.template_match(NewImg, self.refImgforMatch[modelID] )

                    # ==========================
                    # correlated_tracker Method

                    self.tracker[modelID].update(NewImg)
                    tarBox = self.tracker[modelID].get_position()
                    top_left = (tarBox[0],tarBox[1])
                    bottom_right = (tarBox[2],tarBox[3])

                    cv2.rectangle(NewImg, top_left, bottom_right,
                                  (255,50*modelID,50*modelID), 2)

                    print ('tracking_mode')
                    print 'Processing : ( Model : {}, Frame : {} )'.format(modelID, self.numFrames)

                    if len(tarBox)>0: # obj been detected
                        assert len(tarBox)==4
                        position = ((tarBox[0]+tarBox[2])/2, (tarBox[0]+tarBox[2])/2)

                        w = tarBox[2]-tarBox[0]
                        h = tarBox[3]-tarBox[1]
                        self.position[modelID].append(position)

                        # output the refImg by objects #
                        self.refImgforMatch[modelID] = self.have_refimg(NewImg,tarBox,3)

                        # if detected, than change the Flag #
                        self.trackingFlag[modelID]=1

                        NewImg = self.motion_feature_extraction(NewImg,self.position[modelID], modelID,(w,h))

                    # =========================================================
                    # if it is the template match it would happen non-detected
                    else: # Treat as Stationary
                        if len(self.position[modelID])>0:
                            self.position[modelID].append(self.position[modelID][-1])

                        # if detected, than change the Flag
                        self.trackingFlag[modelID]=0
                        NewImg = self.motion_feature_extraction(NewImg,self.position[modelID], modelID)

                    #========================================================
                    # Rollb back Method
                    # if keeping stationary rollback to Ananlyze again
                    # Setting the roll back frame-length could be tricky
                    #
                    if len(self.motionALL[modelID])>rollback :
                        if self.motionALL[modelID][-rollback:]==[5]*rollback:
                            self.trackingFlag[modelID]=0

                            #===========
                            # Roll Back
                            rbFrame = int(rollback/1.5)
                            self.numFrames-=rbFrame*frameInterval
                            self.motionALL[modelID]=self.motionALL[:-rbFrame]
                            # append 0 for breaking the recog loop

                            imgSeq=imgSeq[:-rbFrame]
                            self.position[modelID]=self.position[modelID][:-rbFrame]
                            self.motionLikehoodSeq[modelID]=self.motionLikehoodSeq[modelID][:-rbFrame]

                            # re-init the tracker
                            self.tracker[modelID] = Tracker()
                            break

                self.numFrames+=frameInterval
                imgSeq.append(NewImg)
                gc.collect()
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

        self.motionLikehoodSeq = []
        self.refImgforMatch = []
        for i in range(self.modelNum):
            self.position.append([])
            self.motionALL.append([])

            self.motionLikehoodSeq.append([])
            self.refImgforMatch.append([])
        self.pathRegStr = "None"
        self.actionStr = "None"

    def feature_describe(self):
        return self.motionLikehoodSeq