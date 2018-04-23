__author__ = 'Kent Chiu'

from common_tool_agent.common_func import non_max_suppression
from common_tool_agent.conf import Conf

from common_tool_agent.descriptor_agent.hog import HOG
from common_tool_agent.detect import ObjectDetector

from sklearn.externals import joblib
from skimage.io import imread
from common_tool_agent.common_func import auto_resized
import argparse 
import numpy as np
import cv2
import os


def get_args():
    '''use for single py'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description=''' use -conf to train model-detectors''')
    # Add arguments
    parser.add_argument('-conf', "--conf_path", help="Path to conf_hub",
            required=True)
    parser.add_argument('-img', "--img", help="Path to img",
            required=True)    
    # Parses
    args = parser.parse_args()
    # Assign args to variables
    conf_path = args.conf_path
    img_path = args.img
    # Return all variable values
    return conf_path, img_path


def main():
    conf_path, img_path = get_args()
    conf = Conf(conf_path)
    raw_img = auto_resized(imread(img_path, as_grey=True),conf['train_resolution'])
    # ROI by crowd 
    img = raw_img[20:180,35:345]

    # loading model
    clf = joblib.load(conf['model_ph'])

    # initialize feature container
    hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
    cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])

    # initialize the object detector
    od = ObjectDetector(clf, hog)

    # detect objects in the image and apply non-maxima suppression to the bounding boxes
    (boxes, probs) = od.detect(img, winStep=conf["step_size"], winDim=conf["sliding_size"],
    pyramidScale=conf["pyramid_scale"], minProb=conf["min_probability"])

    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    orig = raw_img.copy()

    # loop over the original bounding boxes and draw them
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)

    # loop over the allowed bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)


if __name__=='__main__':
    main()
    # show the output images
    cv2.imshow("Original", orig)
    cv2.imshow("Image", image)
    cv2.waitKey(0)






