import cv2
 hog = cv2.HOGDescriptor()
 im = cv2.imread(sample)
 h = hog.compute(im)



 writer = cv2.VideoWriter('test1.avi', cv2.VideoWriter_fourcc(*'PIM1'), 25, (640, 480), False)
