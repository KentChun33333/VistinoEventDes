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
from HUB_dictionary.gesture_dictionary import Gestrue_DTW_Dictionary

from HUB_Model.multi_recog import Multi_Model_Iterative_Detect, HaarCV_Recognizor, PureScrewDriverRecog

class Recog2Track():
	def __init__(self, mulRecog, vidObj):
		''' mulRecog = [M1, M2, ...] '''
		self.numFrames = 0
		self.objNumber = len(mulRecog) 


		self.position_values = np.array([])
		self.motionALL = [] # OutPutModel...
		self.motionSeq = []
		self.motionLikehoodSeq = []
		self.motionDictionary = Motion_Dictionary()
		self.pathDictionary = Path_DTW_Dictionary()
		self.pathRegStr = "None" 
		self.actionStr = "None"
		self.recogModels = mulRecog
		self.vidObj = vidObj

	def seq_clip(self, inPut,seqLen):
		if len(inPut)<seqLen:
			return inPut
		return inPut[-seqLen:]

	def perform_analysis(self):
		'''vidObj based on the image io OBJ'''
		while self.numFrames< self.vidObj.get_length():
			img = self.vidObj.get_data(self.numFrames)
			for model in self.recogModels:
				img, featurePosition = 
