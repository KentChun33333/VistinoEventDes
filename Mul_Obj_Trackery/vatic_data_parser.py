#==============================================================================
# Author : Kent Chiu (kentchun33333@gmail.com)
#==============================================================================
# Notice
# - This recipe is used to parser the labeled data from vatic
# -----------------------------------------------------------------------------
# Mode Selection
# - Mode A = Extraction with positive image
# - Mode B = Extraction with negative image
# -----------------------------------------------------------------------------
# Data Description
# 'track_id' : ' All rows with the same ID belong to the same path.',
# 'xmin'     : ' The top left x-coordinate of the bounding box.',
# 'ymin'     : ' The top left y-coordinate of the bounding box.',
# 'xmax'     : ' The bottom right x-coordinate of the bounding box.',
# 'ymax'     : ' The bottom right y-coordinate of the bounding box.',
# 'frame'    : ' The frame that this annotation represents.',
# 'lost'     : ' If 1, the annotation is outside of the view screen.',
# 'occluded' : ' If 1, the annotation is occluded.',
# 'generated': ' If 1, the annotation was automatically interpolated.',
# 'label'    : ' label for this annotation, enclosed in quotation marks.    '
# -----------------------------------------------------------------------------
# Usage Example
# -----------------------------------------------------------------------------
# python vatic_data_parser.py -f id2/example_off_withSave4.txt
#        -v id2/output.avi -o test_id2
#
#==============================================================================

import argparse
import cv2
import imageio
import pandas as pd
from skimage.io import imsave
import os

#==============================================================================
# Get the command-line variables

parser = argparse.ArgumentParser()
parser.add_argument('-f','--file', help='the labeled file path')
parser.add_argument('-v','--vidPath', help='corresponding video path')
parser.add_argument('-o','--output', help='output folder')
arg = vars(parser.parse_args())

#==============================================================================
# make sure output folder is not existed

try:
    assert os.path.isdir(arg['output'])==False
except:
    print ('the output folder is existed')
os.makedirs(arg['output'])

#==============================================================================
# Get the video & labeled data

vid = imageio.get_reader(arg['vidPath'])

with open(arg['file'],'r') as f :
    data = f.readlines()

data2 = [i.split(' ') for i in data]

col = ['track_id','xmin','ymin','xmax','ymax',
       'frameid','lost','occluded','generated','label_name']

df = pd.DataFrame(data2,columns=col )

#==============================================================================
# Hard Mining

def gen_hard_mining(vid, df):
    for frameID in range(vid.get_length()):
        frame = vid.get_data(frameID)
        # get the meta data
        df_tmp = df[ df['frameid']==str(frameID) ]
        df_tmp = df_tmp[['xmin','ymin','xmax','ymax']]
        for i_list in df_tmp.values:
            startX, startY, endX, endY = map(int,i_list)
            cv2.rectangle(frame, (startX,startY), (endX,endY),
                          (255,255,255), -1)
        yield frame

#==============================================================================
# Positive Mining

def positive_img(vid, df, label):
    for frameID in range(vid.get_length()):
        frame = vid.get_data(frameID)
        # get the meta data
        df_tmp = df[ df['frameid']==str(frameID) ]
        df_tmp = df_tmp[ df_tmp['label_name']==label ]
        df_tmp = df_tmp[['xmin','ymin','xmax','ymax']]
        for i_list in df_tmp.values:
            startX, startY, endX, endY = map(int,i_list)
            img = frame[startY:endY, startX:endX,:]
        yield img

if __name__ == '__main__':
    for _, img in enumerate(gen_hard_mining(vid, df)):
        tarFolder = arg['output']+'\\BG\\'
        if not os.path.isdir(tarFolder):
            os.makedirs(tarFolder)
        imsave(tarFolder+str(_)+'.png',img)

    labels = list(set(df['label_name']))
    for label in labels:
        for _, img in enumerate(positive_img(vid, df, label)):
            label_folder = label.replace('"',"").split('\n')[0]
            tarFolder = arg['output']+'\\{}\\'.format(label_folder)

            if not os.path.isdir(tarFolder):
                os.makedirs(tarFolder)
            imsave(tarFolder+str(_)+'.png', img)
