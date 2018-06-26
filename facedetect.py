# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 11:58:54 2018

@author: lenovo
"""

import cv2
import numpy as np
img=cv2.imread("1234.jpg")

face_casecade=cv2.CascadeClassifier(r"C:\Users\lenovo\Desktop\Machine_Learning-master\Machine_Learning-master\haarcascade_frontalface_default.xml")
eye_casecade=cv2.CascadeClassifier(r"C:\Users\lenovo\Desktop\Machine_Learning-master\Machine_Learning-master\haarcascade_eye.xml")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=face_casecade.detectMultiScale(gray,1.3,5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray=gray[y:y+h,x:x+w]
    roi_color=img[y:y+h,x:x+w]
    eyes=eye_casecade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
            
cv2.imshow('img',img)


#cv2.imshow("color",myimage)
#cv2.imshow("gray",gray)
cv2.waitKey(0)
cv2.destroyAllWindows()