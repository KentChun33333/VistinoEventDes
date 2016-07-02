





###########################################################################
# JUST USE ... in the object-detector                                     #                                            
###########################################################################
# python extract-features.py                                              #
# -p ~/MIT_Vedio/2D_DataSet/RHand                                         #
# -n ~/MIT_Vedio/2D_DataSet/BackGround                                    #
###########################################################################







from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
# To read file names
import argparse as ap
import glob
import os



# set HOG parameters 
min_wdw_sz= [100, 40]
step_size= [10, 10]
orientations= 9
pixels_per_cell= [8, 8]
cells_per_block= [3, 3]
visualize= True
normalize= True


im = imread(im_path, as_grey=True)
fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)


