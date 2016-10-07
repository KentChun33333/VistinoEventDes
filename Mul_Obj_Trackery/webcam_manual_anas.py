
#==============================================================================
# Author : Kent Chiu (kentchun33333@gmail.com)
#==============================================================================
# To use this script you have to have to use
# 1. webcam_record.py to record your vid
# 2. webcam_check_initBox.py to check your init tarBox

import argparse
import cv2
import imageio
from HUB_Model.multi_recog import MaunallLable
from com_func.video_io import video_saving_IO,video_saving_CV
from model_Mot2Reg_v2 import Recog2TrackIN

parser = argparse.ArgumentParser()

# Add the video path
parser.add_argument('-i', '--input', required=True,
                    help='input vid path')
parser.add_argument('-o','--output', required=True,
        help='output the vid_path Name & logging Name')
# Add the init bounding box position
parser.add_argument('-b','--tarBox',required=True, type=int, nargs='+')

parser.add_argument('-n', '--objName')
args = vars(parser.parse_args())



vid = imageio.get_reader(args['input'])
a,b,c,d = args['tarBox']
tarBox = (a,b,c,d)

if __name__=='__main__':
    model_1 = MaunallLable(tarBox)

    main_model = Recog2TrackIN([model_1],[args['objName']])
    output = main_model.perform_VID_analysis(1,112,vid,rollback=245)
    print ('All data is in ouput')
    video_saving_IO('{}.avi'.format(args['output']),output)
    print ('Finish Saving')

# python manual_anas.py -i test_1006.avi -o test_1906_AN.avi -n Hand -b 0 270 150 450


# python manual_anas.py -i test_1006_02.avi -o test_1006_02_AN -n Hand -b  230 220 430 450


# python manual_anas.py -i test_1006_03.avi -b 260 230 445 410 -n Hand -o test_1006_03AN

# python manual_anas.py -i test_1006_04.avi -b 110 170 345 405 -n Racket -o test_1006_04AN
