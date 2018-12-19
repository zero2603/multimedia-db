import numpy as np
import cv2
import os

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# count file number in folder frames
list = os.listdir('./frames')
number_files = len(list)

# array to store similarity of 2 consecutive frames
similarity = []

#threshold
T = 0.5

# open file to write result
file = open("result.txt", "w")

index_params = dict(algorithm = 0, trees = 5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

for i in range(number_files-1):

    img1 = cv2.imread('./frames/frame%d.jpg' %i, 0) 
    img2 = cv2.imread('./frames/frame%d.jpg' %(i+1), 0)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
 
    matches = flann.knnMatch(des1, des2, k=2)
    print(i)
    # Apply ratio test
    if len(matches):
        good = []
        for m,n in matches:
            if m.distance < 0.6*n.distance:
                good.append(m)

        avg = (len(kp1) + len(kp2)) / 2
        if avg:
            ratio = len(good) / float(avg)
        else:
            ratio = 0
    else:
        ratio = 0
    
    similarity.append(ratio)

n = len(similarity)

for i in range(1, n-1):
    if similarity[i] < similarity[i-1] and similarity[i] < similarity[i+1]:
        t = i-1
        r = i+1
        while similarity[t] < similarity[t-1]: t = t-1
        while similarity[r] < similarity[r-1]: t = r+1
        
        if similarity[i] < similarity[t]*T or similarity[i] < similarity[r]*T: 
            file.write(str(i) + "\n")

file.close()