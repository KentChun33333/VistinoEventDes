
from skimage.feature import hog
from conf import Conf
from imutils import paths
from sklearn.externals import joblib
from common_func import auto_resized
from skimage.io import imread
import progressbar
import argparse 
import cv2
import os

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


# loopover distraction_dir and slinding_window
def extract(conf):
    # grab the set of ground-truth images and select a percentage of them for training
    trnPaths = list(paths.list_images(conf["pos_raw_ph"]))
    assert len(trnPaths)!=0
    #
    if not os.path.isdir(conf['pos_feat_ph']):
        os.makedirs(conf['pos_feat_ph'])

    # setup the progress bar
    widgets = ["Extracting: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(trnPaths), widgets=widgets).start()
    for (i, imgPath) in enumerate(trnPaths):
        # img = imread(imgPath)
        # resize to the training resolution (wid,height)
        img = auto_resized(imread(imgPath, as_grey=True),conf['sliding_size'])
        fd=hog(img, conf['orientations'], conf['pixels_per_cell'], 
                    conf['cells_per_block'], conf['visualize'], conf['normalize'])
        fd_name = os.path.split(imgPath)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(conf['pos_feat_ph'], fd_name)
        joblib.dump(fd, fd_path)
        pbar.update(i)
    pbar.finish()
    print '[*] Finished Pos.'


def main():
    extract(Conf(get_args()))

if __name__=='__main__':
    main()
