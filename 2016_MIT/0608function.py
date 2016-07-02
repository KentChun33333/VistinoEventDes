import cv2

def turn_rgb(imgIn):
    result = cv2.cvtColor(imgIn,cv2.COLOR_GRAY2BGR)
    return result



# import the necessary packages
import cv2

def update(image_sequence):
	# if the background model is None, initialize it
	bg = image_sequence[0].copy().astype("float")
	for image in image_sequence:
		bg = cv2.accumulateWeighted(image, bg, 0.5)
	return bg

bg = update(result_img[:2])
delta = cv2.absdiff(bg.astype("uint8"), result_img[10])
thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
class MotionDetector:
	def __init__(self, accumWeight=0.5):
		# store the accumulated weight factor
		self.accumWeight = accumWeight

		# initialize the background model
		self.bg = None

	def update(self, image):
		# if the background model is None, initialize it
		if self.bg is None:
			self.bg = image.copy().astype("float")
			return

		# update the background model by accumulating the weighted average
		cv2.accumulateWeighted(image, self.bg, self.accumWeight)

	def detect(self, image, tVal=25):
		# compute the absolute difference between the background model and the image
		# passed in, then threshold the delta image
		delta = cv2.absdiff(self.bg.astype("uint8"), image)
		thresh = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]

		# find contours in the thresholded image
		#(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# in OpenCV3 = des, cnts, hierachy 
		_,cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# if no contours were found, return None
		if len(cnts) == 0:
			return None
		print cnts
		# otherwise, return a tuple of the thresholded image along with the contour area
		return (thresh, max(cnts, key=cv2.contourArea))