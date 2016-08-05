

import cv2, os, imageio
from progress.bar import Bar
from skimage.io import imread, imshow
import time 



class AnnotatingVedioObject():

	def __init__(self, filePath):
		self.filePath = filePath
		self.cap = imageio.get_reader(filePath)
		self._height, self._width, self._channels = self.cap.get_data(0).shape
		self._maxFrameID = self.cap.get_length()-1

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


	def saveVideo_CV(self, fileName, fps, StartFrameID, EndFrameID, Label_ID=True):
		if os.name == 'nt': 
			fourcc = cv2.VideoWriter_fourcc(*'mp4v') # MJPG, mp4v ...etc
			assert(fileName.split('.')[-1]=='avi')
		else: 
			fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
		out = cv2.VideoWriter(fileName, fourcc, fps, (self._width, self._height)) 
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

	def saveVideo_IO(self, fileName, StartFrameID, EndFrameID, Label_ID=True):
		assert(fileName.split('.')[-1]=='avi')
		writer = imageio.get_writer(fileName)
		bar = Bar('Processing', max=(EndFrameID - StartFrameID)) 
		for i in range(StartFrameID, EndFrameID):
			img = self.cap.get_data(i)
			if Label_ID:
				cv2.rectangle(img,(0,0),(350,75),(0,0,0),-1)
				cv2.putText(img, 'FrameId({})'.format(str(i)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
			writer.append_data(img) # Write out frame to video  
			bar.next()
		bar.finish()
		writer.close()
		print ('[*] Finish Saving {} at {}'.format(fileName, os.getcwd()))
