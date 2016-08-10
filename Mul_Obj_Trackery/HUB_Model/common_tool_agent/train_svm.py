

# env lib
from skimage.feature import local_binary_pattern
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import argparse 
import glob
import os
# self lib
from conf import Conf

def get_args():
    '''use for single py must in the main folder Auto_Model'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description=''' use -conf to train model-detectors''')
    # Add arguments
    parser.add_argument('-conf', "--conf_path", help="Path to conf_hub",
            required=True)
    # Parses
    args = parser.parse_args()
    # Assign args to variables
    conf_path = args.conf_path
    # Return all variable values
    return conf_path

def training(conf):
    # Load the positive features
    labels=[]
    fds= []
    for feat_path in glob.glob(os.path.join(conf['pos_feat_ph'],"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)

    # Load the negative features
    for feat_path in glob.glob(os.path.join(conf['neg_feat_ph'],"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)
    #clf = LinearSVC()
    clf = SVC(kernel="linear", C=conf['C'], probability=True, random_state=42)
    print "Training a Linear SVM Classifier"
    clf.fit(fds, labels)
    # If feature directories don't exist, create them
    #if not os.path.isdir(conf['model_ph']):
    #   os.makedirs(conf['model_ph'])
    joblib.dump(clf, conf['model_ph'])
    print "Classifier saved to {}".format(conf['model_ph'])


def main():
    training(Conf(get_args()))

if __name__=='__main__':
    main()