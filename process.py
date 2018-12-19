import numpy as np
import cv2
import os

cap = cv2.VideoCapture('test.mp4')

success, image = cap.read()
count = 0
success = True

while success: 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("./frames/frame%d.jpg" %count, image)
    success, image = cap.read()
    count+=1
    
print("ok")