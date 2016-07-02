


def main_testing_conf():
    '''cd /Users/kentchiu/Night_Graden/Project/2016_MIT/Auto_HOG_SVM/'''
    from common_tool_agent.common_func import non_max_suppression
    from common_tool_agent.conf import Conf
    from common_tool_agent.descriptor_agent.hog import HOG
    from common_tool_agent.detect import ObjectDetector
    from sklearn.externals import joblib
    from skimage.io import imread
    from skimage.io import imshow as show
    from common_tool_agent.common_func import auto_resized
    import argparse 
    import numpy as np
    import cv2
    import os
    import imageio
    conf = Conf('conf_hub/conf_depth_L1_finger_3.json')
    #conf = Conf('conf_hub/conf_001.json')
    #vid = imageio.get_reader('/Users/kentchiu/MIT_Vedio/2016-01-21/10.167.10.158_01_20160121082638418_2.mp4')
    # loading model
    clf = joblib.load(conf['model_ph'])

    # initialize feature container
    hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
    cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
    # initialize the object detector
    od = ObjectDetector(clf, hog)


def video_saving(fileName, fps, imgSequence):
    height, width, channels = imgSequence[0].shape    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(fileName, fourcc, fps, (width, height))    
    for image in imgSequence:
        out.write(image) # Write out frame to video    
    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

def dep_video_saving(fileName, fps, imgSequence):
    height, width = imgSequence[0].shape    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(fileName, fourcc, fps, (width, height))    
    for image in imgSequence:
        img = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB) # avoid 3|4 ERROR
        out.write(img) # Write out frame to video    
    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()


##############################################
#---------------------Depth Tests

def test_with_pro(frame_id,pro):
    ''' Use this fuc when ### above are first implement'''
    img = vid.get_data(frame_id)
    img = auto_resized(img,conf['train_resolution'])
    img_gray = cv2.cvtColor( img , cv2.COLOR_BGR2GRAY)
    roi = img_gray[20:180,35:345]
    (boxes, probs) = od.detect(roi, winStep=conf["step_size"], winDim=conf["sliding_size"],
    pyramidScale=1.1, minProb=pro)
    # since for size effect with the camera, pyramidScale = 1.001, mnust>1, 
    # if positive size would change, we have to use 1.5 or 2 ...etc 
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    orig = img.copy()
    # loop over the allowed bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(orig, (startX+35, startY+20), (endX+35, endY+20), (0, 255, 0), 2)
        print (startX+35, startY+20), (endX+35, endY+20)
    show(orig)
    return img_gray, orig




from skimage.feature import hog
fd = hog(aaa, 9, conf.pixels_per_cell, conf.cells_per_block, True, True)


#=======TEST DEpth 2


def test_with_pro_depth(rawImg,pro):
    ''' Use this fuc when ### above are first implement'''
    #img = vid.get_data(frame_id)
    roi = auto_resized(rawImg,conf['train_resolution'])
    (boxes, probs) = od.detect(roi, winStep=conf["step_size"], winDim=conf["sliding_size"],
    pyramidScale=1.1, minProb=pro)
    # since for size effect with the camera, pyramidScale = 1.001, mnust>1, 
    # if positive size would change, we have to use 1.5 or 2 ...etc 
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    orig = roi.copy()
    # loop over the allowed bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        print (startX, startY), (endX, endY)
    show(orig)
    return roi, orig


def test_with_pro_depth_size(rawImg,pro):
    ''' Use this fuc when ### above are first implement'''
    #img = vid.get_data(frame_id)
    roi = auto_resized(rawImg,conf['train_resolution'])
    (boxes, probs) = od.detect(roi, winStep=conf["step_size"], winDim=conf["sliding_size"],
    pyramidScale=1.1, minProb=pro)
    # since for size effect with the camera, pyramidScale = 1.001, mnust>1, 
    # if positive size would change, we have to use 1.5 or 2 ...etc 
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    orig = roi.copy()
    # loop over the allowed bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        if endX-startX>30:
            pass
        else:
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        print (startX, startY), (endX, endY)
    show(orig)
    return roi, orig


def test_with_pro_depth_iter(rawImg,pro):
    ''' Use this fuc when ### above are first implement'''
    #img = vid.get_data(frame_id)
    roi = auto_resized(rawImg,conf['train_resolution'])
    def memo(proThresh):
        (boxes, probs) = od.detect(roi, winStep=conf["step_size"], winDim=conf["sliding_size"],
        pyramidScale=1.1, minProb=proThresh)
        # since for size effect with the camera, pyramidScale = 1.001, mnust>1, 
        # if positive size would change, we have to use 1.5 or 2 ...etc 
        pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
        if pick>1:
            proThresh+=0.01
            memo(proThresh)
        elif pick ==0:
            proThresh-=0.01
            memo(proThresh)
        else:
            return pick
    pick = memo(0.5)
    orig = img.copy()
    # loop over the allowed bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(orig, (startX+35, startY+20), (endX+35, endY+20), (0, 255, 0), 2)
        print (startX+35, startY+20), (endX+35, endY+20)
    show(orig)
    return roi, orig


list_dir = os.listdir('/Users/kentchiu/MIT_Vedio/3D_DataSet/L_1_finger/')[1:]
result_img=[]
for i in range(70,1000):
    a,b = test_with_pro_depth(imread('/Users/kentchiu/MIT_Vedio/3D_DataSet/L_1_finger/'+list_dir[i]),0.3)
    result_img.append(b)



dep_video_saving('conf_depth_L1_finger_3.mp4',10.0,result_img)

############
# Color Map
############
import cv2 
 
im_gray = cv2.imread("pluto.jpg", cv2.IMREAD_GRAYSCALE)
im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)

# BGR -> RGB
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)




#########
# Keras #
#########


from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])



