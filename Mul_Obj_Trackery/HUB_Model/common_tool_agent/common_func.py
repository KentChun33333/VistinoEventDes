# import the necessary packages
import numpy as np
import imutils
import cv2

def crop_ct101_bb(image, bb, padding=10, dstSize=(32, 32)):
	# unpack the bounding box, extract the ROI from teh image, while taking into account
	# the supplied offset
	(y, h, x, w) = bb
	(x, y) = (max(x - padding, 0), max(y - padding, 0))
	roi = image[y:h + padding, x:w + padding]

	# resize the ROI to the desired destination size
	roi = cv2.resize(roi, dstSize, interpolation=cv2.INTER_AREA)

	# return the ROI
	return roi

def auto_resized(img,size):
    '''size = (width,height)'''
    size = tuple(size)
    resize_img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return resize_img
###########################################################################
# Interpolation                                                           #
###########################################################################
# methods = [                                                             #
#     ("cv2.INTER_NEAREST", cv2.INTER_NEAREST),                           #
#     ("cv2.INTER_LINEAR", cv2.INTER_LINEAR),                             #
#     ("cv2.INTER_AREA", cv2.INTER_AREA),                                 #
#     ("cv2.INTER_CUBIC", cv2.INTER_CUBIC),                               #
#     ("cv2.INTER_LANCZOS4", cv2.INTER_LANCZOS4)]                         #
###########################################################################

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def non_max_suppression(boxes, probs, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes are integers, convert them to floats -- this is important since
	# we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes and sort the bounding boxes by their associated
	# probabilities
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(probs)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value to the list of
		# picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of the bounding box and the
		# smallest (x, y) coordinates for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have overlap greater than the
		# provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked
	return boxes[pick].astype("int")

def delect_non_overlap(boxes1,boxes2, probs, overlapThresh):
	'''
	boxes1: from model 1 or data source 1 (ex) RBG
	boxes2: from model 2 or data source 2 (ex) Depth
	'''
	# if there are no boxes, return an empty list
	if len(boxes1) | len(boxes2)  == 0:
		return []
	# if the bounding boxes are integers, convert them to floats -- this is important since
	# we'll be doing a bunch of divisions
	boxes1 = boxes1.astype("float")
	boxes2 = boxes2.astype("float")
	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes and sort the bounding boxes by their associated
	# probabilities
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(probs)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value to the list of
		# picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of the bounding box and the
		# smallest (x, y) coordinates for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have overlap greater than the
		# provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked
	return boxes[pick].astype("int")

