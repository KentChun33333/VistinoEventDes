__Author__='Kent Chiu'
__CreateDate__ = '2016-08-08'
# This Script is based on the previous version from Motion_Recognition

import numpy as np
import argparse
import imutils
import cv2
import motion_interpretation
from scipy.stats.mstats import mode as statics_mode

from HUB_dictionary.motion_dictionary  import Motion_Dictionary
from HUB_dictionary.path_dictionary    import Path_DTW_Dictionary
from HUB_dictionary.gesture_dictionary import Gesture_DTW_Dictionary
from HUB_Model.multi_recog import Multi_Model_Iterative_Detect, HaarCV_Recognizor, PureScrewDriverRecog

from com_func.conf import Conf

class Recog2Track():
	def __init__(self, mulRecog, modelNameList):
		''' mulRecog = [M1, M2, ...]             
		    modelNameList = [str(M1), str(M2)...] '''
		self.modelNameList = modelNameList
		self.modelNum = len(mulRecog)
		##########################################
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
		###########################################
		self.pathRegStr = "None" 
		self.actionStr = "None"
		self.recogModels = mulRecog

	def perform_IMG_analysis(self, img):
		raw = img.copy()
		for model in self.recogModels:
			raw, tarBox = model.detect(raw)
			if len(tarBox)>0:
				position = ((tarBox[0]+tarBox[2])/2, (tarBox[1]+tarBox[3])/2)
			else :
				position = (0,0)
		return raw, position

	def tracking_mode(self,):
		pass

	def motion_feature_extraction(self, imgAFrecog ,pointsSet,modelID):
		'''None 0 pointsSet'''
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
				"The is %s : %s" % (modelName,motionStr),
				 (20, 50*(1+modelID)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 7)
			# This part spared by not doing path Recognition, 
			# if we want to do path recog, please ref 2016MIT/Motion_Recognition
		return imgAFrecog

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
			imgSeq.append(NewImg)
		self.pathRegStr = 'None'
		return imgSeq

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
class Dispach_Agent_Static_to_Dynamic():
	def __init__():
		pass


PureScrewDriverRecog(Conf('C:/Users/kentc/Documents/GitHub/VistinoEventDes/2016_MIT/Auto_HOG_SVM/conf_hub/conf_pureScrewDriver_2.json'))
model.showDetect(vid.get_data(305))