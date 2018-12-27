import numpy as np
import cv2
import os
from moviepy.editor import *

N = 1
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# count file number in folder frames
list = os.listdir('./frames')
number_files = len(list)

# array to store similarity of 2 consecutive frames
similarity = []

boundaries = []

keypoints = []

#threshold
T = 0.5

# open file to write result
# file = open("result.txt", "w")

index_params = dict(algorithm = 0, trees = 5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)
# bf = cv2.BFMatcher()

# run loop
for i in range(0, number_files-N-1, N):

    img1 = cv2.imread('./frames/frame%d.jpg' %i, 0) 
    img2 = cv2.imread('./frames/frame%d.jpg' %(i+N), 0)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    if(len(keypoints) == 0):
        keypoints.append(kp1)
        keypoints.append(kp2)
    else:
        keypoints.append(kp2)
 
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

for i in range(1, n-2):
    if similarity[i] < similarity[i-1] and similarity[i] < similarity[i+1]:
        t = i-1
        r = i+1
        while similarity[t] < similarity[t-1]: t = t-1
        if r < n-2:
            while similarity[r] < similarity[r+1]: r = r+1
        
        if similarity[i] < similarity[t]*T or similarity[i] < similarity[r]*T: 
            # file.write(str(i) + "\n")
            boundaries.append(i)
            
# file.close()
video = VideoFileClip("test.mp4")
for i in range (len(boundaries)-2):
    clip_start = int(boundaries[i]) * N / float(25)
    clip_end = int(boundaries[i+1]) * N / float(25)
    clip = video.subclip(clip_start, clip_end)
    clip.write_videofile("./output/shot_%s.mp4" %i)