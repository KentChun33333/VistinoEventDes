    




###########################################################################
# Usage Note @ Kent (Jin-Chun CHiu)                                       #  
###########################################################################
# 1. Doing Operation in /bin                                              #
# 2. Be aware of resize of image, because of the                          #
#    'ValueError: setting an array element with a sequence'               #
#    Above Error would happen due to the np.ndarray is not the same size  #
###########################################################################



# Import the required modules
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import argparse as ap
import glob
import os
import numpy as np

model_path = '../data/models/svm0505.model'

clf_type = "LIN_SVM"

pos_feat_path =  '../data/features/pos'
neg_feat_path = '../data/features/neg'
fds = []
labels = []
# Load the positive features
for feat_path in glob.glob(os.path.join(pos_feat_path,"*.feat")):
    fd = joblib.load(feat_path)
    fds.append(fd)
    labels.append(1)

# Load the negative features
for feat_path in glob.glob(os.path.join(neg_feat_path,"*.feat")):
    fd = joblib.load(feat_path)
    fds.append(fd)
    labels.append(0)

if clf_type is "LIN_SVM":
    clf = LinearSVC()
    print "Training a Linear SVM Classifier"
    clf.fit(fds, labels)
    # If feature directories don't exist, create them
    if not os.path.isdir(os.path.split(model_path)[0]):
        os.makedirs(os.path.split(model_path)[0])
    joblib.dump(clf, model_path)
    print "Classifier saved to {}".format(model_path)





