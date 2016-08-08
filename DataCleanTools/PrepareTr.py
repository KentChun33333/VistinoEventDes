__Author__ = 'KentChiu'
__Update__ = '2016-08-05'

import glob, os, imageio, cv2
import argprase as arg
from json_conf import Conf
#from sklearn.externals.joblib import Parallel, delayed
from sklearn.externals import joblib


def ConfJSON():
    parser = arg.ArgumentParser(description='use -conf to prepare training data')
    parser.add_argument('-conf', "--conf_path", help="Path of conf.json",
            required=True)
    args = parser.parse_args() # Parses
    return Conf(args.conf_path)


folderList = confJSON['folderList']

def TrainFilePath(folderList, constrain=None, **kargs ):
	if constrain is None:
		constrain = ('avi', 'mp4')
	for basePath in folderList :
		for (rootDir, dirNames, fileNames) in os.walk(basePath):
			for fileName in fileNames:
				if fileName.split('.')[-1] in constrain:
					yield os.path.join(rootDir, fileName), rootDir

# glob.glob(os.path.join(conf['pos_feat_ph'],"*.feat"))

# For Feature Extraction then IO

######################
# For Training Part ##
######################
##  ##
######
####
####




TrX = []
TrY = []


model_1 = Haar_Recognizor()
model_2 = test_with_pro_2



multiModel = MultiModel([model1, model2])



for vidPath, rootDir in TrainFilePath(folderList):
	vid = imageio.get_reader(vidPath)
	for i in range(vid.get_length()):
		frame = vid.get_data(i)
		TrX.append = model.extract(frame)
		TrY.append = rootDir


#(1) Minsubstraction (50~75 frames) 3 seconds

####################
# For Detect Part ##
####################
##  ##
######
####
####


# filesList = os.listdir(confJSON['folderName'])
	filesList = os.walk(confJSON['folderName'])
	filesList = [s for s in filesList if s.split('.')[-1] in ('avi', 'mp4')]
	os.path.join()



