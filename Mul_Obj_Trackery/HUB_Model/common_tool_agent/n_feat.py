from descriptor_agent.hog import HOG
from conf import Conf
from skimage.feature import hog
from imutils import paths
from sklearn.externals import joblib
from skimage.io import imread
from common_func import auto_resized
import progressbar
import argparse 
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
    # Parses
    args = parser.parse_args()
    # Assign args to variables
    conf_path = args.conf_path
    # Return all variable values
    return conf_path

def sliding_window(image, stepSize, winSize):
    # slide a window across the image
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            # yield the current window
                yield (x, y, image[y:y + winSize[1], x:x + winSize[0]])

# loopover distraction_dir and slinding_window
def extract(conf):
    # grab the set of ground-truth images and select a percentage of them for training
    trnPaths = list(paths.list_images(conf["neg_raw_ph"]))
    assert len(trnPaths)!=0
    #
    if not os.path.isdir(conf['neg_feat_ph']):
        os.makedirs(conf['neg_feat_ph'])
    # setup the progress bar
    widgets = ["Extracting: ", progressbar.Percentage(), " ",
        progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(trnPaths), widgets=widgets).start()
    for (i, imgPath) in enumerate(trnPaths):
        # img = imread(imgPath)
        # resize to the training resolution (wid,height)
        img = auto_resized(imread(imgPath, as_grey=True),conf['train_resolution'])
        #######################################################
        # if some pos_img in neg_img with stationary position #
        #######################################################
        if conf['special_cover'] is not None:
            print '[*] Triggerd special_cover'
            startX,startY,endX,endY = conf['special_cover']
            cv2.rectangle(img, (startX-10,startY-10), (endX+10,endY+10), 0, -1) 
            #################################
            # -1 = filled, 0 = outline only #
            #################################
        fd_id=0 #just a index for recording feature_file
        for x,y,window in sliding_window(img, conf['step_size'],conf['sliding_size']):
            if window.shape != tuple(conf['sliding_size']):
                pass
            else :
                fd=hog(window, conf['orientations'], conf['pixels_per_cell'], 
                    conf['cells_per_block'], conf['visualize'], conf['normalize'])
                fd_name = os.path.split(imgPath)[1].split(".")[0] +str(fd_id)+ ".feat"
                fd_path = os.path.join(conf['neg_feat_ph'], fd_name)
                joblib.dump(fd, fd_path)
                fd_id+=1
        pbar.update(i)
    pbar.finish()
    print '[*] Finished Neg.'



def main():
    extract(Conf(get_args()))

if __name__=='__main__':
    main()



