# Import the functions to calculate feature descriptors
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
import cv2
# To read file names
import argparse 
import glob
import os
from config import *
from test-classifier import sliding_window

###########################################################################
# Interpolation                                                           #
###########################################################################
# methods = [                                                             #
#     ("cv2.INTER_NEAREST", cv2.INTER_NEAREST),                           #
#     ("cv2.INTER_LINEAR", cv2.INTER_LINEAR),                             #
#     ("cv2.INTER_AREA", cv2.INTER_AREA),                                 #
#     ("cv2.INTER_CUBIC", cv2.INTER_CUBIC),                               #
#     ("cv2.INTER_LANCZOS4", cv2.INTER_LANCZOS4)]                         #
###########################################################################


def get_args():
    '''This function parses and return arguments passed in'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description=''' Script read both depth and rgb frames from kinetic 
        and save them in Assigned Folder''')
    # Add arguments
    parser.add_argument('-p', "--pospath", help="Path to positive images",
            required=True)
    parser.add_argument('-n', "--negpath", help="Path to negative images",
            required=True)
    parser.add_argument('-d', "--descriptor", help="Use - HOG",
            default="HOG")
    # the HOGw sould corresponding to the min_window of sliding wondow ...?
    parser.add_argument('-HOGw',"--width", help="img_width to HOG",default=240)
    parser.add_argument('-HOGh',"--height", help="img_height to HOG",default=120)
    args = parser.parse_args()
    ####################################
    # args = vars(parser.parse_args()) #
    # pos_im_path = args["pospath"]    #
    ####################################

    # Assign args to variables
    pos_im_path = args.pospath
    neg_im_path = args.negpath
    des_type = args.descriptor
    width, height = args.width , args.height
    # Return all variable values
    return pos_im_path, neg_im_path, des_type, width, height

def resized(img,w,h):
    resize = cv2.resize(img, (w,h), interpolation=cv2.INTER_CUBIC)
    return resize

def main():
    # Accquire Argument Parsers
    pos_im_path, neg_im_path, des_type, width, height = get_args()

    # If feature directories don't exist, create them
    if not os.path.isdir(pos_feat_ph):
        os.makedirs(pos_feat_ph)

    # If feature directories don't exist, create them
    if not os.path.isdir(neg_feat_ph):
        os.makedirs(neg_feat_ph)

    print "Calculating the descriptors for the positive samples and saving them"
    for im_path in glob.glob(os.path.join(pos_im_path, "*")):
        # Add below becasuse of HC training need .txt file by Kent
        if im_path.split('.')[-1]!='txt':
            im = resized(imread(im_path, as_grey=True),width,height)
            if des_type == "HOG":
                fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
            fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
            fd_path = os.path.join(pos_feat_ph, fd_name)
            joblib.dump(fd, fd_path)
    print "Positive features saved in {}".format(pos_feat_ph)
    #
    print "Calculating the descriptors for the negative samples and saving them"

    for im_path in glob.glob(os.path.join(neg_im_path, "*")):
        # Add below because of HC training need .txt file by Kent
        if im_path.split('.')[-1]!='txt':
            im = resized(imread(im_path, as_grey=True),width,height)
            if des_type == "HOG":
                fd = hog(im,  orientations, pixels_per_cell, cells_per_block, visualize, normalize)
            fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
            fd_path = os.path.join(neg_feat_ph, fd_name)
            joblib.dump(fd, fd_path)
    print "Negative features saved in {}".format(neg_feat_ph)

    print "Completed calculating features from training images"


if __name__ == "__main__":
    main()
