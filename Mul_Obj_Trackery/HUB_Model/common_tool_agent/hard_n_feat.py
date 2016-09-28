from descriptor_agent.hog import HOG
from conf import Conf
from skimage.feature import hog as _hog
from imutils import paths
from sklearn.externals import joblib
from skimage.io import imread
from common_func import auto_resized
import progressbar
import argparse
import cv2
import os
from datetime import datetime

def get_args():
    '''use for single py'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description=''' use -conf to train model-detectors''')
    # Add arguments
    parser.add_argument('-conf', "--conf_path", help="Path to conf_hub",
            required=True)
    parser.add_argument('-hn', "--hn_path", help="Path to hard-img",
            required=True)
    # Parses
    args = parser.parse_args()
    # Assign args to variables
    conf_path = args.conf_path
    hn_path = args.hn_path
    # Return all variable values
    return conf_path, hn_path

def hard_neg_extrect(conf,imgPath):
    '''gray_input image and str(fd.name)'''
    img = auto_resized(imread(imgPath, as_grey=True),conf['sliding_size'])

    # if need , use below :
    # img = auto_resized(img ,conf['sliding_size'])
    from datetime import datetime
    from skimage.feature import hog as _hog
    fd=_hog(img, conf['orientations'], conf['pixels_per_cell'],
                conf['cells_per_block'], conf['visualize'], conf['normalize'])

    fd_name = str(datetime.now()).split(':')[0].replace(' ','').replace('-','')
    fd_path = os.path.join(conf['neg_feat_ph'], fd_name)
    for i in range(2000):
        joblib.dump(fd, fd_path+str(i)+'.feat')

def hard_neg_extrect_with_img(img,conf):


def main():
    conf_path, hn_path = get_args()
    hard_neg_extrect(Conf(conf_path),hn_path)
    return

if __name__=="__main__":
    main()
