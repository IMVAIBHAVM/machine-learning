# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 11:16:27 2018

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 11:58:54 2018

@author: lenovo
"""

import cv2
import numpy as np
img=cv2.imread("8956326.jpg")

face_casecade=cv2.CascadeClassifier(r"C:\Users\lenovo\Desktop\Machine_Learning-master\Machine_Learning-master\haarcascade_russian_plate_number.xml")

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=face_casecade.detectMultiScale(gray,1.02,5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray=gray[y:y+h,x:x+w]
    roi_color=img[y:y+h,x:x+w]
   
    
            
cv2.imshow('img',img)


#cv2.imshow("color",myimage)
#cv2.imshow("gray",gray)
cv2.waitKey(0)
cv2.destroyAllWindows()