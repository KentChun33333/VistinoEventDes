__Author__ = 'Kent Chiu'

from common_tool_agent.common_func import non_max_suppression
from common_tool_agent.conf import Conf
from common_tool_agent.descriptor_agent.hog import HOG
from common_tool_agent.detect import ObjectDetector
from skimage.io import imread
from skimage.io import imshow as show
from sklearn.externals import joblib
from common_tool_agent.common_func import auto_resized
import numpy as np
import cv2

class CallBack(object):
    def __init__(self):
        self.staticmodel = StaticModel()

class StaticModel(object):
    def __init__(self):
        pass

class HaarCV_Recognizor(StaticModel):
    def __init__(self, xmlPath='model_hub/opencv_cascade/Rhand_no_tools/cascade.xml'):
        # store the number of orientations, pixels per cell, cells per block, and
        # whether normalization should be applied to the image
        self.xmlPath=xmlPath

    def detect(self, img):
        hand_5 = cv2.CascadeClassifier(self.xmlPath)
        ref = img.copy()
        hand5 = hand_5.detectMultiScale(
            ref.copy()[50:550,50:800],
            scaleFactor=1.2, minNeighbors=15, #35 ==> 1
            #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            )
        hand_roi=[]
        for (x,y,w,h) in hand5:
        # Position Aware + Size Aware
            if x < 180 or y<180 or h<90 :
                pass
            else:
            # draw rectangle at the specific location
                cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 
                cv2.putText(ref, "Hand", (x+50,y+50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)
                hand_roi.append([x+50,y+50,x+50+w,y+50+h])
        if len(hand_roi)>1:
            tarBox = max(hand_roi) # Chose One
        elif len(hand_roi)==1:
            tarBox = hand_roi.pop()
        else:
            tarBox = []
        return ref, tarBox # if None = []

    def show_detect(self, img):
        show(self.detect(img)[0])

    def recog_handposition(self, img):
        newImg, box = self.detect(img)
        x = int((box[0]+box[2])*0.5)
        y = int((box[1]+box[3])*0.5)
        return (x,y)


class PureScrewDriverRecog(StaticModel):
    def __init__(self, conf):
        '''Ex: conf = Conf('/Users/kentchiu/VistinoEventDes/2016_MIT/Auto_HOG_SVM/conf_hub/conf_pureScrewDriver_2.json')'''
        self.conf = conf

    def detect(self, rawImg, pro=0.7, scale=1.3):
        hog = HOG(orientations=self.conf["orientations"], pixelsPerCell=tuple(self.conf["pixels_per_cell"]),
        cellsPerBlock=tuple(self.conf["cells_per_block"]), normalize=self.conf["normalize"])
        # initialize the object detector
        clf = joblib.load(self.conf['model_ph'])
        od = ObjectDetector(clf, hog)
        ref = rawImg.copy()
        img = auto_resized(ref,self.conf['train_resolution'])
        img_gray = cv2.cvtColor( img , cv2.COLOR_BGR2GRAY)
        roi = img_gray
        (boxes, probs) = od.detect(roi, winStep=self.conf["step_size"], winDim=self.conf["sliding_size"],
        pyramidScale=scale, minProb=pro)
        # since for size effect with the camera, pyramidScale = 1.001, mnust>1, 
        # if positive size would change, we have to use 1.5 or 2 ...etc 
        tarBox = non_max_suppression(np.array(boxes), probs, self.conf["overlap_thresh"])
        orig = img.copy()    

        # Resize Back, I am GOD !!!!! 
        y_sizeFactor = ref.shape[0]/float(img.shape[0])
        x_sizeFactor = ref.shape[1]/float(img.shape[1])    

        # loop over the allowed bounding boxes and draw them
        tarBoxSet=[]
        for (startX, startY, endX, endY) in tarBox:
            #cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
            startX = int(startX* x_sizeFactor)
            endX   = int(endX  * x_sizeFactor)
            startY = int(startY* y_sizeFactor)
            endY   = int(endY  * y_sizeFactor)    
            if startX < 300 or startY > 250 or startX > 400 or (endX -startX)>200:
                continue    
            cv2.rectangle(ref, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(ref, "SkrewDriver", (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            #print (startX, startY), (endX, endY)
            tarBoxSet.append([startX, startY, endX, endY])
        # Refine the output obj to only one!
        if len(tarBoxSet)>1:
            tarBox = max(tarBoxSet) # Chose One
        elif len(tarBoxSet)==1:
            tarBox = tarBoxSet.pop()
        else:
            tarBox = [] # if no obj detect , retuen []
        return ref, tarBox

    def show_detect(self, img, pro=0.6, scale=1.3):
        show(self.detect(img,pro, scale)[0])

class SKin_Hand_Detection(StaticModel):
    def __init__(self, Flag = False):
        self.path = 'model_hub/opencv_cascade/frontalFace10/haarcascade_frontalface_alt.xml'
        self.Flag = Flag
        if Flag is True:
            path = self.path
            model_1 = HaarCV_Recognizor(path)

    def usage(self):
        print '''
        model = SKin_Hand_Detection()
        Recog2Track
        '''

    def face_Remove(self, frame):
        face_cascade = cv2.CascadeClassifier(self.path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,
            scaleFactor=1.2,
            minNeighbors=10, #35 ==> 1
            ) 
        # Detect the face and remove it from hand detection 
        for (x,y,w,h) in faces:
            tarBox = (x,y,x+w, y+h)
            cv2.rectangle(frame,(tarBox[0],tarBox[1]),(tarBox[2],tarBox[3]),(10,10,10),-1)
        if self.Flag is True:
        # Detect the face and remove it from hand detection with Other Model 
            _, tarBox = model_1.detect(clone)
            if len(tarBox)==4:
                cv2.rectangle(frame,(tarBox[0],tarBox[1]),(tarBox[2],tarBox[3]),(10,10,10),-1)
        return frame

    def skin_thresh_contours(self,frame):
        '''
        imput the frame, 
        return thresh & contours 
        '''

        # blut the image
        blur = cv2.blur(frame,(3,3))    
        
        #Convert to HSV color space
        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
        
        #Create a binary image with where white will be skin colors and rest is black
        mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
        
        #Kernel matrices for morphological transformation    
        kernel_square = np.ones((11,11),np.uint8)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        
        #Perform morphological transformations to filter out the background noise
        #Dilation increase skin color area
        #Erosion increase skin color area
        dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
        erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
        dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
        filtered = cv2.medianBlur(dilation2,5)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
        dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
        median = cv2.medianBlur(dilation2,5)
        ret,thresh = cv2.threshold(median,127,255,0)
        
        #Find contours of the filtered frame    

        #_, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
        if int(cv2.__version__[0])==2:
            (contours, _) = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else :
            (_,contours, _) = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 

        return thresh, contours


    def detect(self, frame):
        '''
        return labeled img and bounding box as tarBox
        it could be used in the Recog2Track model
        '''
        # Add Clone for Hand Detection Operation 
        # The clone img is not draw by solid rectangle or ...etc 
        clone = frame.copy()     
        
        # Remove the Face Skin Region 
        frame = self.face_Remove(frame)
        # cv2.imshow("Face Remove : Press ESC to EXIT", frame)

        # find the skin thresh image and its contours
        thresh, contours = self.skin_thresh_contours(frame)

        
        #Find Max contour area (Assume that hand is in the frame)
        max_area=100
        ci=0    
        for i in range(len(contours)):
            cnt=contours[i]
            area = cv2.contourArea(cnt)
            if(area>max_area):
                max_area=area
                ci=i  
                
        #Largest area contour
        if len(contours)==0:
            tarBox = [] # if None return []
            return clone, tarBox

        cnts = contours[ci]    

        #Find convex hull
        hull = cv2.convexHull(cnts)
        
        #Find convex defects
        hull2 = cv2.convexHull(cnts,returnPoints = False)
        defects = cv2.convexityDefects(cnts,hull2)
        
        #Get defect points and draw them in the original image
        FarDefect = []
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnts[s][0])
            end = tuple(cnts[e][0])
            far = tuple(cnts[f][0])
            FarDefect.append(far)
            cv2.line(clone,start,end,[0,255,0],1)
            cv2.circle(clone,far,10,[100,255,255],3)
        
        #Find moments of the largest contour
        moments = cv2.moments(cnts)
        
        #Central mass of first order moments
        if moments['m00']!=0:
            cx = int(moments['m10']/moments['m00']) # cx = M10/M00
            cy = int(moments['m01']/moments['m00']) # cy = M01/M00
        centerMass=(cx,cy)    
        
        #Draw center mass
        cv2.circle(clone,centerMass,7,[100,0,255],2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(clone,'Center',tuple(centerMass),font,2,(255,255,255),2)     
        
        #Distance from each finger defect(finger webbing) to the center mass
        distanceBetweenDefectsToCenter = []
        for i in range(0,len(FarDefect)):
            x =  np.array(FarDefect[i])
            centerMass = np.array(centerMass)
            distance = np.sqrt(np.power(x[0]-centerMass[0],2)+np.power(x[1]-centerMass[1],2))
            distanceBetweenDefectsToCenter.append(distance)
        
        #Get an average of three shortest distances from finger webbing to center mass
        sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
        AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])
     
        #Get fingertip points from contour hull
        #If points are in proximity of 80 pixels, consider as a single point in the group
        finger = []
        for i in range(0,len(hull)-1):
            if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
                if hull[i][0][1] <500 :
                    finger.append(hull[i][0])
        
        #The fingertip points are 5 hull points with largest y coordinates  
        finger =  sorted(finger,key=lambda x: x[1])   
        fingers = finger[0:5]
        
        #Calculate distance of each finger tip to the center mass
        fingerDistance = []
        for i in range(0,len(fingers)):
            distance = np.sqrt(np.power(fingers[i][0]-centerMass[0],2)+np.power(fingers[i][1]-centerMass[0],2))
            fingerDistance.append(distance)
        
        #Finger is pointed/raised if the distance of between fingertip to the center mass is larger
        #than the distance of average finger webbing to center mass by 130 pixels
        result = 0
        for i in range(0,len(fingers)):
            if fingerDistance[i] > AverageDefectDistance+130:
                result = result +1

        
        # Print bounding rectangle
        x,y,w,h = cv2.boundingRect(cnts)
        img = cv2.rectangle(clone,(x,y),(x+w,y+h),(0,255,0),2)
        
        cv2.drawContours(clone,[hull],-1,(255,255,255),2)
        # bounding box recontruction in order to fit into the Recog2Track()
        tarBox = (cx-10, cy-10, cx+10, cy+10)
        return clone, tarBox



class ConvNetFactory:
    def __init__(self):
        pass

    @staticmethod
    def build(name, *args, **kargs):
        # define the network (i.e., string => function) mappings
        mappings = {
            "shallownet": ConvNetFactory.ShallowNet,
            "lenet": ConvNetFactory.LeNet,
            "karpathynet": ConvNetFactory.KarpathyNet,
            "minivggnet": ConvNetFactory.MiniVGGNet}

        # grab the builder function from the mappings dictionary
        builder = mappings.get(name, None)

        # if the builder is None, then there is not a function that can be used
        # to build to the network, so return None
        if builder is None:
            return None

        # otherwise, build the network architecture
        return builder(*args, **kargs)

    @staticmethod
    def ShallowNet(numChannels, imgRows, imgCols, numClasses, **kwargs):
        # initialzie the model
        model = Sequential()

        # define the first (and only) CONV => RELU layer
        model.add(Convolution2D(32, 3, 3, border_mode="same",
            input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation("relu"))

        # add a FC layer followed by the soft-max classifier
        model.add(Flatten())
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))

        # return the network architecture
        return model

    @staticmethod
    def LeNet(numChannels, imgRows, imgCols, numClasses, activation="tanh", **kwargs):
        # initialize the model
        model = Sequential()

        # define the first set of CONV => ACTIVATION => POOL layers
        model.add(Convolution2D(20, 5, 5, border_mode="same",
            input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # define the second set of CONV => ACTIVATION => POOL layers
        model.add(Convolution2D(50, 5, 5, border_mode="same"))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # define the first FC => ACTIVATION layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))

        # define the second FC layer
        model.add(Dense(numClasses))

        # lastly, define the soft-max classifier
        model.add(Activation("softmax"))

        # return the network architecture
        return model

    @staticmethod
    def KarpathyNet(numChannels, imgRows, imgCols, numClasses, dropout=False, **kwargs):
        # initialize the model
        model = Sequential()

        # define the first set of CONV => RELU => POOL layers
        model.add(Convolution2D(16, 5, 5, border_mode="same",
            input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # check to see if dropout should be applied to reduce overfitting
        if dropout:
            model.add(Dropout(0.25))

        # define the second set of CONV => RELU => POOL layers
        model.add(Convolution2D(20, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # check to see if dropout should be applied to reduce overfitting
        if dropout:
            model.add(Dropout(0.25))

        # define the third set of CONV => RELU => POOL layers
        model.add(Convolution2D(20, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # check to see if dropout should be applied to reduce overfitting
        if dropout:
            model.add(Dropout(0.5))

        # define the soft-max classifier
        model.add(Flatten())
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))

        # return the network architecture
        return model

    @staticmethod
    def MiniVGGNet(numChannels, imgRows, imgCols, numClasses, dropout=False, **kwargs):
        # initialize the model
        model = Sequential()

        # define the first set of CONV => RELU => CONV => RELU => POOL layers
        model.add(Convolution2D(32, 3, 3, border_mode="same",
            input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation("relu"))
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # check to see if dropout should be applied to reduce overfitting
        if dropout:
            model.add(Dropout(0.25))

        # define the second set of CONV => RELU => CONV => RELU => POOL layers
        model.add(Convolution2D(64, 3, 3, border_mode="same"))
        model.add(Activation("relu"))
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # check to see if dropout should be applied to reduce overfitting
        if dropout:
            model.add(Dropout(0.25))

        # define the set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))

        # check to see if dropout should be applied to reduce overfitting
        if dropout:
            model.add(Dropout(0.5))

        # define the soft-max classifier
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))

        # return the network architecture
        return model

'Picking up Screw Driver'
'Putting down Screw Driver'
'The Hand is moving with out holding something'
'The hand is holding Scrw Driver doing something'

# model_1 = HaarCV_Recognizor()
# model_2 = PureScrewDriverRecog(Conf('conf_hub/conf_pureScrewDriver_2.json'))
# Multi_Test=Multi_Model_Iterative_Detect([model_2, model_1])
# Multi_Test.showDetect(vid.get_data(305))

#Model_2 = PureScrewDriverRecog(Conf('C:/Users/kentc/Documents/GitHub/VistinoEventDes/2016_MIT/Auto_HOG_SVM/conf_hub/conf_pureScrewDriver_2.json'))
#model = Multi_Model_Iterative_Detect([Model_1,Model_2])#

#model.showDetect(vid.get_data(12))#

#show(Model_2.detect(vid.get_data(300), pro = 0.6)[0])
#show(Model_2.detect(vid.get_data(300), pro = 0.8)[0])
#show(Model_2.detect(vid.get_data(300), pro = 0.85)[0])
#model = Multi_Model_Iterative_Detect([Model_1,Model_2])
#model.showDetect(vid.get_data(305))
#model.showDetect(vid.get_data(315))
#model.showDetect(vid.get_data(1))
#model.showDetect(vid.get_data(12))