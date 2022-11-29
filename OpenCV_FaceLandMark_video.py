#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 13:05:55 2022

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
import os


#%%
Path_image_root = "example_face/"
Video_name = "AFTERNOON.mp4"
Path_VideoImageExample_root = "example_lip_extracted/VideoImageExample/"
Path_fullFrames = "frameExtraction/"
Path_2dFrames = "Lip_frameByFrame/"
Path_3dMatrixs = "Lip_3dMatrix/"
#%%

def Video2Frames(Path_video, Path_fullFrame, Path_2dFrame, Path_3dMatrix):
    '''
    Parameters
    ----------
    Path_video : String
        DESCRIPTION: directories which the video is stored
    Path_fullFrame : String
        DESCRIPTION: diretories which the decomposed fullFrames of the video is stored
    Path_2dFrame: String 
        DESCRIPTION: diretories which the extracted lip information are stored as individual frames
    Path_3dMatrix: String
        DESCRIPTION: directories which the extracted lip information are stored as 3d matrix (concatenated along the timeline)

    Returns
    -------
    None.
    '''


    vidcap = cv2.VideoCapture(Path_video)
    count = 0
    while True:
        success,image = vidcap.read()
        if not success:
            break
        else: 
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image_rgb)
            plt.show()
            cv2.imwrite(os.path.join(Path_fullFrame,"frame{:d}.jpg".format(count)), image_rgb)     # save frame as JPEG file
            count += 1
    print("{} images are extacted in {}.".format(count,Path_fullFrame))


def FacialLandmark(image):
    '''
    This method extracts the 68 Facial landmarks from a given front face image
    indexs that are on the lips: 48-67
    Parameters
    ----------
    image : np.array                    (256,256,3)
        image matrix

    Returns
    -------
    landmarks: np.array                     (1,68,2)
        landmark detected from image

    '''
    plt.imshow(image)
    plt.show()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(image_gray)
    plt.show()
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
    return landmark


# @TODO: implement Type I, Type II, Type III, and Type IV 
# Type I 
# Type II 
# Type III 

    

#%%
######################################################################################
#                                       Test field                                   #
######################################################################################

#%% FaceLandmark Test 
Example_image = cv2.imread(Path_VideoImageExample_root+Path_fullFrames+'frame0.jpg')
landmark = FacialLandmark(Example_image)

#%% itergration test
Video2Frames(Path_image_root+Video_name, Path_VideoImageExample_root+Path_fullFrames, Path_VideoImageExample_root+Path_2dFrames, Path_VideoImageExample_root+Path_3dMatrixs)