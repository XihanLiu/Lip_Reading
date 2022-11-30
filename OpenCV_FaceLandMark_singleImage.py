#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:20:34 2022

@author: mikewang
"""
# opencv 4.1.2 to read images
import cv2
# used for accessing url to download files
import urllib.request as urlreq
# used to access local directory
import os
# used to plot our images
import matplotlib.pyplot as plt
# used to change image size
from pylab import rcParams
import numpy as np
from scipy.io import savemat

#%%
# Path_image = "example_face/1.jpeg"
Path_image = "example_lip_extracted/GIFImageExample/frameExtraction/ABOUT_00002_31.gif/frame2.jpg"
Path_lip = "example_lip_extracted/SingleImageExample/"
# read image with openCV
image = cv2.imread(Path_image)
# plot image with matplotlib package
plt.imshow(image)

#%%
# convert image to RGB colour
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plot image with matplotlib package
plt.imshow(image_rgb)
plt.show()

# set dimension for cropping image
x, y, width, depth = 0, 0, 500, 300
image_cropped = image_rgb[y:(y+depth), x:(x+width)]

# create a copy of the cropped image to be used later
image_template = image_cropped.copy()

# convert image to Grayscale
image_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

# remove axes and show image
plt.axis("off")
plt.imshow(image_gray, cmap = "gray")
plt.show()


#%%
# save face detection algorithm's url in haarcascade_url variable
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"

# save face detection algorithm's name as haarcascade
haarcascade = "haarcascade_frontalface_alt2.xml"

# chech if file is in working directory
if (haarcascade in os.listdir(os.curdir)):
    print("File exists")
else:
    # download file from url and save locally as haarcascade_frontalface_alt2.xml, < 1MB
    urlreq.urlretrieve(haarcascade_url, haarcascade)
    print("File downloaded")

# create an instance of the Face Detection Cascade Classifier
detector = cv2.CascadeClassifier(haarcascade)

# Detect faces using the haarcascade classifier on the "grayscale image"
faces = detector.detectMultiScale(image_gray)

# Print coordinates of detected faces
print("Faces:\n", faces)

for face in faces:
#     save the coordinates in x, y, w, d variables
    (x,y,w,d) = face
    # Draw a white coloured rectangle around each face using the face's coordinates
    # on the "image_template" with the thickness of 2 
    cv2.rectangle(image_template,(x,y),(x+w, y+d),(255, 255, 255), 2)

plt.axis("off")
plt.imshow(image_template)
plt.title('Face Detection')
plt.show()

#%%
# save facial landmark detection model's url in LBFmodel_url variable
LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

# save facial landmark detection model's name as LBFmodel
LBFmodel = "lbfmodel.yaml"

# check if file is in working directory
if (LBFmodel in os.listdir(os.curdir)):
    print("File exists")
else:
    # download picture from url and save locally as lbfmodel.yaml, < 54MB
    urlreq.urlretrieve(LBFmodel_url, LBFmodel)
    print("File downloaded")

# create an instance of the Facial landmark Detector with the model
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)

# Detect landmarks on "image_gray"
_, landmarks = landmark_detector.fit(image_gray, faces)
landmark = landmarks[0]
for landmark in landmarks:
    for x,y in landmark[0]:
		# display landmarks on "image_cropped"
		# with white colour in BGR and thickness 1
        cv2.circle(image_cropped, (int(x), int(y)), 1, (255, 255, 255), 1)
plt.axis("off")
plt.imshow(image_cropped)
plt.show()

#%% Extract rectangle image containing the mouth
###################################################################
#                             Type I                              #
###################################################################
'''
key points in landmark matrix: 
    x_min = landmark[0,4,0]
    x_max = landmark[0,12,0]
    y_min = landmark[0,33,1]
    y_max = landmark[0,5,1]
'''
x_min = int(landmark[0,4,0])
x_max = int(landmark[0,12,0])
y_min = int(landmark[0,33,1])
y_max = int(landmark[0,5,1])
standard_dimension = (100,50) # first x then y 
small_dimension = (50,25)

# set dimension for cropping image
x, y, width, depth = x_min, y_min, x_max-x_min, y_max-y_min,
lip_rect_o = image_rgb[y:(y+depth), x:(x+width)]
lip_rect = cv2.resize(lip_rect_o, standard_dimension)
lip_rect = cv2.cvtColor(lip_rect, cv2.COLOR_BGR2RGB)
cv2.imwrite(Path_lip + "Type1_standard.jpg",lip_rect)
plt.axis("off")
plt.imshow(lip_rect)
plt.show()

lip_rect_small = cv2.resize(lip_rect_o, small_dimension)
lip_rect_small = cv2.cvtColor(lip_rect_small, cv2.COLOR_BGR2RGB)
cv2.imwrite(Path_lip + "Type1_small.jpg",lip_rect_small)
plt.axis("off")
plt.imshow(lip_rect_small)
plt.show()

#%% Extract lip only croped image
'''
indexs that are on the lips:
    48-67
'''
pts_all = landmark[0,48:67,:]
pts_outter = landmark[0,48:59,:]


## Crop the bounding rect
rect = cv2.boundingRect(pts_all)
x,y,w,h = rect
croped = image_rgb[y:y+h, x:x+w].copy()
plt.axis("off")
plt.imshow(croped)
plt.show()

###################################################################
#                             Type II                             #
###################################################################

## make mask on the lip
pts_all = pts_all - pts_all.min(axis=0)

mask = np.zeros(croped.shape[:2], np.uint8)
ctr = np.array(pts_all).reshape((-1,1,2)).astype(np.int32)
cv2.drawContours(mask, [ctr], -1, (255, 255, 255), -1, cv2.LINE_AA)
lip_mask = cv2.resize(mask, standard_dimension)
lip_mask = cv2.cvtColor(lip_mask, cv2.COLOR_BGR2RGB)
cv2.imwrite(Path_lip + "Type2_standard.jpg",lip_mask)
plt.axis("off")
plt.imshow(lip_mask)
plt.show()

lip_mask_small = cv2.resize(mask, small_dimension)
lip_mask_small = cv2.cvtColor(lip_mask_small, cv2.COLOR_BGR2RGB)
cv2.imwrite(Path_lip + "Type2_small.jpg",lip_mask_small)
plt.axis("off")
plt.imshow(lip_mask_small)
plt.show()



###################################################################
#                             Type III                            #
###################################################################

## make mask on the outter counter
pts_outter = pts_outter - pts_outter.min(axis=0)

mask = np.zeros(croped.shape[:2], np.uint8)
ctr = np.array(pts_outter).reshape((-1,1,2)).astype(np.int32)
cv2.drawContours(mask, [ctr], -1, (255, 255, 255), -1, cv2.LINE_AA)


## (3) do bit-op
dst = cv2.bitwise_and(croped, croped, mask=mask)
lip_dst = cv2.resize(dst, standard_dimension)
lip_dst = cv2.cvtColor(lip_dst, cv2.COLOR_BGR2RGB)
cv2.imwrite(Path_lip + "Type3_standard.jpg",lip_dst)
plt.axis("off")
plt.imshow(lip_dst)
plt.show()

lip_dst_small = cv2.resize(dst, small_dimension)
lip_dst_small = cv2.cvtColor(lip_dst_small, cv2.COLOR_BGR2RGB)
cv2.imwrite(Path_lip + "Type3_small.jpg",lip_dst_small)
plt.axis("off")
plt.imshow(lip_dst_small)
plt.show()






#%% Extract lip landmark distance matrix
###################################################################
#                             Type IV                             #
###################################################################
'''
indexs that are on the lips:
    48-67
generate relative distance vector
'''
def rotation_once(landmark_array):
    landmark_list = list(landmark_array)
    landmark_list.append(landmark_list.pop(0))
    return np.array(landmark_list)

dist_list = []
numDistForEachNode = landmark[0,:,0].shape[0]-1
landmark_copy = landmark[0,:,:].copy()
landmark_copy = rotation_once(landmark_copy)

for i in range(numDistForEachNode): 
    dist = np.linalg.norm(landmark[0,:,:]-landmark_copy,axis=1)
    dist_list.append(dist)
    landmark_copy = rotation_once(landmark_copy)
    
dist_array = np.array(dist_list)

mdic = {"dist_array": dist_array}
savemat(Path_lip+"Type4.mat",mdic)

    




