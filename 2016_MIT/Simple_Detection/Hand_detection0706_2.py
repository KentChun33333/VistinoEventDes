import cv2
import numpy as np

for image_index in xrange(12, 15):
    frame = cv2.imread("output_convex_hull/" + str(image_index) + "_warped.jpg")

    blur = cv2.blur(frame, (3, 3))
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([15, 255, 255]))

    kernel_square = np.ones((11, 11), np.uint8)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    dilation = cv2.dilate(mask, kernel_ellipse, iterations = 1)
    erosion = cv2.erode(dilation, kernel_square, iterations = 1)

    dilation2 = cv2.dilate(erosion, kernel_ellipse, iterations = 1)

    filtered = cv2.medianBlur(dilation2, 5)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    dilation2 = cv2.dilate(filtered, kernel_ellipse, iterations = 1)

    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation3 = cv2.dilate(filtered, kernel_ellipse, iterations = 1)

    median = cv2.medianBlur(dilation2, 5)
    ret, thresh = cv2.threshold(median, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #Find Max contour area (Assume that hand is in the frame)
    max_area = -1 * float("inf")
    biggest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            biggest_contour = contour

    hull = cv2.convexHull(biggest_contour)

    fingers = []
    for i in xrange(len(hull) - 1):
        if (np.absolute(hull[i][0][0] - hull[i + 1][0][0]) > 50) or (np.absolute(hull[i][0][1] - hull[i + 1][0][1]) > 50):
            fingers.append(hull[i][0])

    # Sort fingers by height
    fingers = sorted(fingers, key=lambda x: x[1])
    cv2.circle(frame, tuple(fingers[0]), 50, (0, 0, 255), -1)
    cv2.imwrite(str(image_index) + ".jpg", frame)