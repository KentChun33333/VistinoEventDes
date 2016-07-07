import cv2
import numpy as np
import time

#face_cascade = cv2.CascadeClassifier(
#    '/Users/kentchiu/opencv/data/haarcascades/haarcascade_frontalcatface.xml')
#eye_cascade = cv2.CascadeClassifier(
#    '/Users/kentchiu/opencv/data/haarcascades/haarcascade_eye.xml')

face_cascade = cv2.CascadeClassifier(
    '/Users/kentchiu/VistinoEventDes/2016_MIT/frontalFace10/haarcascade_frontalface_alt.xml')

eye_cascade = cv2.CascadeClassifier(
    '/Users/kentchiu/VistinoEventDes/2016_MIT/frontalEyes35x16/frontalEyes35x16.xml')

nose_cascade = cv2.CascadeClassifier(
    '/Users/kentchiu/VistinoEventDes/2016_MIT/xml/Nariz.xml')
month_cascade = cv2.CascadeClassifier(
    '/Users/kentchiu/VistinoEventDes/2016_MIT/xml/Mouth.xml')

# Detect faces in the image
cap = cv2.VideoCapture(0)

while(1):
    _, frame = cap.read()
    time.sleep(0.1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray,
        scaleFactor=1.2,
        minNeighbors=15, #35 ==> 1
        )
    for (x,y,w,h) in faces:
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        nose = nose_cascade.detectMultiScale(roi_gray)
        month = month_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            if ey > (y+0.35*h) or ew<0.7*w:
                pass
            else:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                # nose
                for (nx,ny,nw,nh) in nose:
                    if ny<ey+eh*0.75 or ny>(y+0.63*h):
                        pass
                    else:
                        cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,0,0),2)
                        for (mx,my,mw,mh) in month:
                            if my<ny+nh*0.75:
                                pass
                            else:
                                cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(255,255,255),2)
        cv2.imshow('img',img)
    k = cv2.waitKey(3) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()



#-------------------------
#cap = cv2.VideoCapture(0)
#while(1):
#    _, frame = cap.read()
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    #gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#    faces = face_cascade.detectMultiScale(
#        gray,
#        scaleFactor=1.2,
#        minNeighbors=15, #35 ==> 1
#        )
#    for (x,y,w,h) in faces:
#        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
#        roi_gray = gray[y:y+h, x:x+w]
#        eyes = eye_cascade.detectMultiScale(roi_gray)
#        for (ex,ey,ew,eh) in eyes:
#            if ey > 0.5*(y+y+h):
#                pass
#            else:
#                cv2.rectangle(frame,(ex+x,ey+y),(ex+x+ew,ey+y+eh),(0,255,0),2)
#        nose = nose_cascade.detectMultiScale(roi_gray)
#        for (nx,ny,nw,nh) in nose:
#            if ny<ey+eh*0.75 or ny>0.5*(y+y+h):
#                pass
#            else:
#                cv2.rectangle(frame,(nx+x,ny+y),(nx+x+nw,ny+y+nh),(0,0,0),2)
#        month = month_cascade.detectMultiScale(roi_gray)
#        for (mx,my,mw,mh) in month:
#            if my<ny+nh*0.75:
#                pass
#            else:
#                cv2.rectangle(frame,(mx+x,my+y),(mx+x+mw,my+y+mh),(255,255,255),2)
#        cv2.imshow('img',frame)
#    k = cv2.waitKey(3) & 0xFF
#    if k == 27:
#        break
