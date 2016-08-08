
'Picking up Screw Driver'
'Putting down Screw Driver'
'The Hand is moving with out holding something'
'The hand is holding Scrw Driver doing something'

# model_1 = HaarCV_Recognizor()
# model_2 = PureScrewDriverRecog(Conf('conf_hub/conf_pureScrewDriver_2.json'))
# Multi_Test=Multi_Model_Iterative_Detect([model_2, model_1])
# Multi_Test.showDetect(vid.get_data(305))

from common_tool_agent.common_func import non_max_suppression
from common_tool_agent.conf import Conf
from common_tool_agent.descriptor_agent.hog import HOG
from common_tool_agent.detect import ObjectDetector
from skimage.io import imread
from skimage.io import imshow as show
from common_tool_agent.common_func import auto_resized
import numpy as np
import cv2


class Multi_Model_Iterative_Detect:
	def __init__(self , DetectObjectList):
		'''
		DetectObjectList = [ M1, M2, ...]
		Models must have detect attributes
		'''
		self.models = DetectObjectList
	def detect(self, img):
		raw = img.copy()
		for model in self.models:
			raw, position = model.detect(raw)
		return raw
	def showDetect(self,img):
		show(self.detect(img))

class HaarCV_Recognizor:
    def __init__(self, xmlPath='/Users/kentchiu/MIT_Vedio/Rhand_no_tools/cascade.xml'):
        # store the number of orientations, pixels per cell, cells per block, and
        # whether normalization should be applied to the image
        self.xmlPath=xmlPath

    def detect(self, img):
        hand_5 = cv2.CascadeClassifier(self.xmlPath)
        ref = img.copy()
        hand5 = hand_5.detectMultiScale(
            ref.copy()[50:550,50:800],
            scaleFactor=1.2, minNeighbors=15, #35 ==> 1
            #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            )
        hand_roi=[]
        for (x,y,w,h) in hand5:
        # Position Aware + Size Aware
            if (x < 160 and y < 160) or y<160 or h<90:
                pass
            else:
            # draw rectangle at the specific location
                cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 
                cv2.putText(ref, "Hand", (x+50,y+50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)
                hand_roi.append([x+50,y+50,x+50+w,y+50+h])
        if len(hand_roi)>1:
            result_box = max(hand_roi)
        elif len(hand_roi)==1:
            result_box = hand_roi.pop()
        else:
            result_box = 0
        return ref, result_box

    def showDetect(self, img):
        show(self.detect(img)[0])

    def recog_HandPosition(self, img):
        newImg, box = self.detect(img)
        x = int((box[0]+box[2])*0.5)
        y = int((box[1]+box[3])*0.5)
        return (x,y)


class PureScrewDriverRecog:
    def __init__(self, conf):
        '''Ex: conf = Conf('conf_hub/conf_pureScrewDriver_2.json')'''
        self.conf = conf

    def detect(self, rawImg, pro=0.8, showFlag=None):
        hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
        cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
        # initialize the object detector
        od = ObjectDetector(clf, hog)
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
        if showFlag is not None:
            show(ref)
        return ref, pick
    def showDetect(self, img):
        show(self.detect(img)[0])
