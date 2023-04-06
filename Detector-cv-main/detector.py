import numpy as np
import cv2 as cv
import imutils
from collections import deque

lower = {'red':(166, 84, 141), 'green':(66, 122, 129), 'blue':(97, 100, 117), 'yellow':(23, 59, 119), 'orange':(0, 50, 80)}
upper = {'red':(186,255,255), 'green':(86,255,255), 'blue':(117,255,255), 'yellow':(54,255,255), 'orange':(20,255,255)}
colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'yellow':(0, 255, 217), 'orange':(0,140,255)}

video = cv.VideoCapture(0)

while True:
    grabbed , frame = video.read()
    frame = imutils.resize(frame, width=600) 
    blurred = cv.GaussianBlur(frame, (11, 11), 0)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    for key, value in upper.items():
        kernel = np.ones((9,9),np.uint8)
        mask = cv.inRange(hsv, lower[key], upper[key])
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        if len(contours) > 0:
            c = max(contours, key=cv.contourArea)
            ((x, y), radius) = cv.minEnclosingCircle(c)
            M = cv.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if radius > 0:
                cv.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
                cv.putText(frame,key + " ball", (int(x-radius),int(y-radius)), cv.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)


    cv.imshow("mask",mask)
    cv.imshow("webcam",hsv)
    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break

video.release()
cv.destroyAllWindows()