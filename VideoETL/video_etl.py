import cv2, os, imageio
from progress.bar import Bar
from skimage.io import imread, imshow
import time 

def saveVideo_IO(vid, fileName, StartFrameID, EndFrameID, Label_ID=True):
	assert(fileName.split('.')[-1]=='avi')
	writer = imageio.get_writer(fileName)
	bar = Bar('Processing', max=(EndFrameID - StartFrameID)) 
	for i in range(StartFrameID, EndFrameID):
		img = vid.get_data(i)
		if Label_ID:
			cv2.rectangle(img,(0,0),(350,75),(0,0,0),-1)
			cv2.putText(img, 'FrameId({})'.format(str(i)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
		writer.append_data(img) # Write out frame to video  
		bar.next()
	bar.finish()
	writer.close()
	print ('[*] Finish Saving {} at {}'.format(fileName, os.getcwd()))

def saveVideo_IO(ImageSeq, fileName, Label_ID=True):
	assert(fileName.split('.')[-1]=='avi')
	writer = imageio.get_writer(fileName)
	bar = Bar('Processing', max=(EndFrameID - StartFrameID)) 
	for i in range(StartFrameID, EndFrameID):
		img = vid.get_data(i)
		if Label_ID:
			cv2.rectangle(img,(0,0),(350,75),(0,0,0),-1)
			cv2.putText(img, 'FrameId({})'.format(str(i)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
		writer.append_data(img) # Write out frame to video  
		bar.next()
	bar.finish()
	writer.close()
	print ('[*] Finish Saving {} at {}'.format(fileName, os.getcwd()))

def video_saving_IO(fileName, imgSequence):
    assert(fileName.split('.')[-1]=='avi')
    writer = imageio.get_writer(fileName)
    for image in imgSequence:
        writer.append_data(image)
    # Release everything if job is finished
    writer.close()
    print ('[*] Finish Saving {} at {}'.format(fileName, os.pardir.join([os.getcwd(),fileName])))


def video_saving_CV(fileName, fps, imgSequence):
    assert(fileName.split('.')[-1]=='avi')
    height, width, channels = imgSequence[0].shape    
    bar = Bar('Processing', max=len(imgSequence)) 
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(fileName, fourcc, fps, (width, height))    
    for image in imgSequence:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        out.write(image) # Write out frame to video    
    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()