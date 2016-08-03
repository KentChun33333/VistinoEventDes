

import cv2, os, imageio
import imageio
from progress.bar import Bar
from skimage.io import imread, imshow
import time 
class AnnotatingVedioObject():

	def __init__(self, filePath):
		self.filePath = filePath
		self.cap = imageio.get_reader(filePath)
		self.height, self.width, self.channels = self.cap.get_data(0).shape

	def CapRetrival(self, FrameId):
		return self.cap.get_data(FrameId)

	def showFrame(self, FrameId):
		imshow(self.cap.get_data(FrameId))

	def showMovie(self, StartFrameID, EndFrameID, Label_ID=True):
		for i in range(StartFrameID, EndFrameID):
			img = self.cap.get_data(i)
			img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
			if Label_ID:
				cv2.rectangle(img,(0,0),(350,75),(0,0,0),-1)
				cv2.putText(img, 'FrameId({})'.format(str(i)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
			cv2.imshow("Frame", img)
			time.sleep(1)
		time.sleep(20)
		cv2.destroyAllWindows()


	def saveVideo_ID(self, fileName, fps, StartFrameID, EndFrameID, Label_ID=True):
		if os.name == 'nt': 
			fourcc = cv2.VideoWriter_fourcc(*'MJPG')
			assert(fileName.split('.')[-1]=='avi')
		else: 
			fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
		out = cv2.VideoWriter(fileName, fourcc, fps, (self.width, self.height)) 
		bar = Bar('Processing', max=(EndFrameID - StartFrameID)) 
		for i in range(StartFrameID, EndFrameID):
			img = self.cap.get_data(i)
			img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
			if Label_ID:
				cv2.rectangle(img,(0,0),(350,75),(0,0,0),-1)
				cv2.putText(img, 'FrameId({})'.format(str(i)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
			out.write(img) # Write out frame to video  
			bar.next()
		bar.finish()
		out.release()
		print ('[*] Finish Saving {} at {}'.format(fileName, os.getcwd()))

