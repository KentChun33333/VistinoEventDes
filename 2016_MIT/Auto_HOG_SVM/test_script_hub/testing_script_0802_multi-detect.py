
'Grabing the Screw Driver'
'Putting down the Screw Driver'
'The Hand is moving with out holding something'
'The hand is holding Scrw Driver doing something'


from skimage.io import imread, imshow as read, show

def model_dictionnary():
	['hand', 'ScrewDriver']

def multiDetector(img, multiModels):
	raw = img.copy()
	feature_object=[]
	for model in multiModels:
		box, position = model(img.copy)
		feature.append(box)


    conf = Conf('conf_hub/conf_RHand_with_tool_4.json')
    clf = joblib.load(conf['model_ph'])

    # initialize feature container
    hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
    cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
    # initialize the object detector
    od = ObjectDetector(clf, hog)

def test_with_pro_2(rawImg,pro):
    ''' Use this fuc when ### above are first implement'''
    ref = rawImg.copy()
    img = auto_resized(rawImg,conf['train_resolution'])
    img_gray = cv2.cvtColor( img , cv2.COLOR_BGR2GRAY)
    roi = img_gray
    (boxes, probs) = od.detect(roi, winStep=conf["step_size"], winDim=conf["sliding_size"],
    pyramidScale=1.1, minProb=pro)
    # since for size effect with the camera, pyramidScale = 1.001, mnust>1, 
    # if positive size would change, we have to use 1.5 or 2 ...etc 
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    orig = img.copy()

    # Resize Back, I am GOD !!!!! 
    y_sizeFactor = ref.shape[0]/float(img.shape[0])
    x_sizeFactor = ref.shape[1]/float(img.shape[1])

    # loop over the allowed bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        #cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        startX = int(startX* x_sizeFactor)
        endX   = int(endX  * x_sizeFactor)
        startY = int(startY* y_sizeFactor)
        endY   = int(endY  * y_sizeFactor)        
        cv2.rectangle(ref, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(ref, "Hodling SkrewDriver", (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        #print (startX, startY), (endX, endY)
    show(ref)
    return ref, pick
#-------------------------------------------------------------------------
class Haar_Recognizor:
    def __init__(self, xmlPath='/Users/kentchiu/MIT_Vedio/Rhand_no_tools/cascade.xml'):
        # store the number of orientations, pixels per cell, cells per block, and
        # whether normalization should be applied to the image
        self.xmlPath=xmlPath

    def recognise(self, img):
        hand_5 = cv2.CascadeClassifier(self.xmlPath)
        ref = img.copy()
        hand5 = hand_5.detectMultiScale(
            ref.copy()[50:550,50:800],
            scaleFactor=1.2,
            minNeighbors=15, #35 ==> 1
            #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            )
        hand = []
        hand_roi=[]
        for (x,y,w,h) in hand5:
        # Position Aware + Size Aware
            if (x < 160 and y < 160) or y<160 or h<90:
                pass
            else:
            # draw rectangle at the specific location
                cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 
                cv2.putText(ref, "Hand", (x+50,y+50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)
                hand.append([(x+50+x+50+w)/2,(y+50+h+y+50)/2])
        #show(ref)
                print ((x+50,y+50),(x+50+w,y+50+h))
                hand_roi.append([x+50,y+50,x+50+w,y+50+h])
        if len(hand_roi)>1:
            result_box = max(hand_roi)
        elif len(hand_roi)==1:
            result_box = hand_roi.pop()
        else:
            result_box = 0
        return ref, result_box

    def recog_show(self, img):
        hand_5 = cv2.CascadeClassifier(self.xmlPath)
        ref = img.copy()
        hand5 = hand_5.detectMultiScale(
            ref.copy()[50:550,50:800],
            scaleFactor=1.2,
            minNeighbors=15, #35 ==> 1
            #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            )
        hand = []
        hand_roi=[]
        for (x,y,w,h) in hand5:
        # Position Aware + Size Aware
            if (x < 160 and y < 160) or y<160 or h<90:
                pass
            else:
            # draw rectangle at the specific location
                cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 
                cv2.putText(ref, "Hand", (x+50,y+50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)
                hand.append([(x+50+x+50+w)/2,(y+50+h+y+50)/2])
        #show(ref)
                print ((x+50,y+50),(x+50+w,y+50+h))
                hand_roi.append([x+50,y+50,x+50+w,y+50+h])
        if len(hand_roi)>1:
            result_box = max(hand_roi)
        elif len(hand_roi)==1:
            result_box = hand_roi.pop()
        else:
            result_box = 0
        show(ref)

    def recog_HandPosition(self, img):
        hand_5 = cv2.CascadeClassifier(self.xmlPath)
        ref = img.copy()
        hand5 = hand_5.detectMultiScale(
            ref.copy()[50:550,50:800],
            scaleFactor=1.2,
            minNeighbors=15, #35 ==> 1
            #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            )
        hand = []
        hand_roi=[]
        for (x,y,w,h) in hand5:
        # Position Aware + Size Aware
            if (x < 160 and y < 160) or y<160 or h<90:
                pass
            else:
            # draw rectangle at the specific location
                cv2.rectangle(ref,(x+50,y+50),(x+50+w,y+50+h),(255,0,0),2) 
                cv2.putText(ref, "Hand", (x+50,y+50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)
                hand.append([(x+50+x+50+w)/2,(y+50+h+y+50)/2])
        #show(ref)
                print ((x+50,y+50),(x+50+w,y+50+h))
                hand_roi.append([x+50,y+50,x+50+w,y+50+h])
        if len(hand_roi)>1:
            result_box = max(hand_roi)
        elif len(hand_roi)==1:
            result_box = hand_roi.pop()
        else:
            result_box = 0
        return hand

class Single_Object_Recognisor():
	def __init__(self, ):

