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

def Video2Frames(Path_video, Path_3dMatrix, VideoName):
    '''
    Parameters
    ----------
    Path_video : String
        DESCRIPTION: directories which the video is stored
    Path_3dMatrix: String
        DESCRIPTION: directories which the extracted lip information are stored as 3d matrix (concatenated along the timeline)
    VideoName: String
        DESCRIPTION: Video name
    Returns
    -------
    None.
    '''

    # reading the video file
    vidcap = cv2.VideoCapture(Path_video)
    count = 0
    
    #------------------------------------------------------------------------------------#
    #                                making directories                                  #
    #------------------------------------------------------------------------------------#
    # @TODO: make prediction dirctory in the example folder 
    Path_currentVideo_3dMatrix_small_typeIII = Path_3dMatrix+VideoName+"/"+"small/typeIII/"
    os.makedirs(Path_currentVideo_3dMatrix_small_typeIII, exist_ok=True)
    
    lip_dst_small_list = []
    
    # @TODO load trained nerual network model
    
    
    
    while True:
        success,image = vidcap.read()
        if not success:
            
            break
        else: 
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            landmark = FacialLandmark(image_rgb)
            # Type III
            lip_dst, lip_dst_small = typeIII_2dFrame(image_rgb, landmark, count)
            lip_dst_small_list.append(lip_dst_small)

    
    lip_dst_small_3d_array = np.array(lip_dst_small_list)
    mdic = {"lip_dst_small_3d_array": lip_dst_small_3d_array}
    savemat(Path_currentVideo_3dMatrix_small_typeIII+"_3dTypeIII.mat",mdic)
    






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




def typeIII_2dFrame(image_rgb, landmark, Frame_num):
    '''

    Parameters
    ----------
    image_rgb : TYPE
        DESCRIPTION.
    landmark : TYPE
        DESCRIPTION.
    typeII_directory_standard : TYPE
        DESCRIPTION.
    typeII_directory_small : TYPE
        DESCRIPTION.

    Returns
    -------
    lip_dst : TYPE
        DESCRIPTION.
    lip_dst_small : TYPE
        DESCRIPTION.

    '''
    standard_dimension = (100,50) # first x then y 
    small_dimension = (50,25)
    pts_all = landmark[0,48:67,:]
    pts_outter = landmark[0,48:59,:]
    ## Crop the bounding rect
    rect = cv2.boundingRect(pts_all)
    x,y,w,h = rect
    croped = image_rgb[y:y+h, x:x+w].copy()
    ## make mask on the outter counter
    pts_outter = pts_outter - pts_outter.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    ctr = np.array(pts_outter).reshape((-1,1,2)).astype(np.int32)
    cv2.drawContours(mask, [ctr], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    lip_dst = cv2.resize(dst, standard_dimension)
    lip_dst = cv2.cvtColor(lip_dst, cv2.COLOR_BGR2RGB)
    lip_dst_small = cv2.resize(dst, small_dimension)
    lip_dst_small = cv2.cvtColor(lip_dst_small, cv2.COLOR_BGR2RGB)
    return lip_dst, lip_dst_small


    

#%%