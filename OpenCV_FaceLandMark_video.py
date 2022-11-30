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

def Video2Frames(Path_video, Path_fullFrame, Path_2dFrame, Path_3dMatrix, VideoName):
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
    # FULL FRAME PATH
    Path_currentVideoFullFrame = Path_fullFrame+VideoName+"/"
    
    # 2dFRAME PATH
    Path_currentVideo_2dFrame_standard_typeI = Path_2dFrame+VideoName+"/"+"standard/typeI/"
    Path_currentVideo_2dFrame_standard_typeII = Path_2dFrame+VideoName+"/"+"standard/typeII/"
    Path_currentVideo_2dFrame_standard_typeIII = Path_2dFrame+VideoName+"/"+"standard/typeIII/"
    
    
    Path_currentVideo_2dFrame_small_typeI = Path_2dFrame+VideoName+"/"+"small/typeI/"
    Path_currentVideo_2dFrame_small_typeII = Path_2dFrame+VideoName+"/"+"small/typeII/"
    Path_currentVideo_2dFrame_small_typeIII = Path_2dFrame+VideoName+"/"+"small/typeIII/"
    Path_currentVideo_2dFrame_typeIV = Path_2dFrame+VideoName+"/"+"typeIV/"
    
    #3dMatrix PATH
    Path_currentVideo_3dMatrix_standard_typeI = Path_3dMatrix+VideoName+"/"+"standard/typeI/"
    Path_currentVideo_3dMatrix_standard_typeII = Path_3dMatrix+VideoName+"/"+"standard/typeII/"
    Path_currentVideo_3dMatrix_standard_typeIII = Path_3dMatrix+VideoName+"/"+"standard/typeIII/"
    
    Path_currentVideo_3dMatrix_small_typeI = Path_3dMatrix+VideoName+"/"+"small/typeI/"
    Path_currentVideo_3dMatrix_small_typeII = Path_3dMatrix+VideoName+"/"+"small/typeII/"
    Path_currentVideo_3dMatrix_small_typeIII = Path_3dMatrix+VideoName+"/"+"small/typeIII/"
    Path_currentVideo_3dMatrix_typeIV = Path_3dMatrix+VideoName+"/"+"typeIV/"
    
    
    # make new directories
    os.makedirs(Path_currentVideoFullFrame, exist_ok=True)
    
    os.makedirs(Path_currentVideo_2dFrame_standard_typeI, exist_ok=True)
    os.makedirs(Path_currentVideo_2dFrame_standard_typeII, exist_ok=True)
    os.makedirs(Path_currentVideo_2dFrame_standard_typeIII, exist_ok=True)
    os.makedirs(Path_currentVideo_2dFrame_typeIV, exist_ok=True)
    os.makedirs(Path_currentVideo_2dFrame_small_typeI, exist_ok=True)
    os.makedirs(Path_currentVideo_2dFrame_small_typeII, exist_ok=True)
    os.makedirs(Path_currentVideo_2dFrame_small_typeIII, exist_ok=True)
    
    os.makedirs(Path_currentVideo_3dMatrix_standard_typeI, exist_ok=True)
    os.makedirs(Path_currentVideo_3dMatrix_standard_typeII, exist_ok=True)
    os.makedirs(Path_currentVideo_3dMatrix_standard_typeIII, exist_ok=True)
    os.makedirs(Path_currentVideo_3dMatrix_typeIV, exist_ok=True)
    os.makedirs(Path_currentVideo_3dMatrix_small_typeI, exist_ok=True)
    os.makedirs(Path_currentVideo_3dMatrix_small_typeII, exist_ok=True)
    os.makedirs(Path_currentVideo_3dMatrix_small_typeIII, exist_ok=True)
    
    lip_rect_list = []
    lip_mask_list = []
    lip_dst_list = []
    lip_rect_small_list = []
    lip_mask_small_list = []
    lip_dst_small_list = []
    dist_array_list = []
    
    
    
    while True:
        success,image = vidcap.read()
        if not success:
            
            break
        else: 
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # plt.imshow(image_rgb)
            # plt.show()
            # Full Frame seperation
            cv2.imwrite(os.path.join(Path_currentVideoFullFrame,"frame{:d}.jpg".format(count)), image_rgb)     # save frame as JPEG 
            landmark = FacialLandmark(image_rgb)
            # Type I
            lip_rect, lip_rect_small = typeI_2dFrame(image_rgb, landmark, Path_currentVideo_2dFrame_standard_typeI, Path_currentVideo_2dFrame_small_typeI, count)
            lip_rect_list.append(lip_rect)
            lip_rect_small_list.append(lip_rect_small)
            # Type II
            lip_mask, lip_mask_small = typeII_2dFrame(image_rgb, landmark, Path_currentVideo_2dFrame_standard_typeII, Path_currentVideo_2dFrame_small_typeII, count)
            lip_mask_list.append(lip_mask)
            lip_mask_small_list.append(lip_mask_small)
            # Type III
            lip_dst, lip_dst_small = typeIII_2dFrame(image_rgb, landmark, Path_currentVideo_2dFrame_standard_typeIII, Path_currentVideo_2dFrame_small_typeIII, count)
            lip_dst_list.append(lip_dst)
            lip_dst_small_list.append(lip_dst_small)
            # Type IV
            dist_array = typeIV_2dFrame(landmark, Path_currentVideo_2dFrame_typeIV, count)
            dist_array_list.append(dist_array)
            count += 1
    
    # print("{} images are extacted in {}.".format(count,Path_fullFrame))
    lip_rect_3d_array = np.array(lip_rect_list)
    lip_rect_small_3d_array = np.array(lip_rect_small_list)
    mdic = {"lip_rect_3d_array": lip_rect_3d_array}
    savemat(Path_currentVideo_3dMatrix_standard_typeI+"_3dTypeI.mat",mdic)
    mdic = {"lip_rect_small_3d_array": lip_rect_small_3d_array}
    savemat(Path_currentVideo_3dMatrix_small_typeI+"_3dTypeI.mat",mdic)
    
    lip_mask_3d_array = np.array(lip_mask_list)
    lip_mask_small_3d_array = np.array(lip_mask_small_list)
    mdic = {"lip_mask_3d_array": lip_mask_3d_array}
    savemat(Path_currentVideo_3dMatrix_standard_typeII+"_3dTypeII.mat",mdic)
    mdic = {"lip_mask_small_3d_array": lip_mask_small_3d_array}
    savemat(Path_currentVideo_3dMatrix_small_typeII+"_3dTypeII.mat",mdic)
    
    lip_dst_3d_array = np.array(lip_dst_list)
    lip_dst_small_3d_array = np.array(lip_dst_small_list)
    mdic = {"lip_dst_3d_array": lip_dst_3d_array}
    savemat(Path_currentVideo_3dMatrix_standard_typeIII+"_3dTypeIII.mat",mdic)
    mdic = {"lip_dst_small_3d_array": lip_dst_small_3d_array}
    savemat(Path_currentVideo_3dMatrix_small_typeIII+"_3dTypeIII.mat",mdic)
    
    dist_3d_array = np.array(dist_array_list)
    mdic = {"dist_3d_array": dist_3d_array}
    savemat(Path_currentVideo_3dMatrix_typeIV+"_3dTypeIV.mat",mdic)






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


def typeI_2dFrame(image_rgb, landmark, typeI_directory_standard, typeI_directory_small, Frame_num):
      '''
      Parameters
      ----------
      image_rgb : np.array                    (256,256,3)
          image matrix
      landmark : np.array                     (1,68,2)
          landmark detected from image
      typeI_directory : String
          typeI extraction images storing directory

      Returns
      -------
      lip_rect: np.array                        (100,50,3)
      lip_rect_small: np.array                  (50,25,3)

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
      cv2.imwrite(typeI_directory_standard + "Frame"+str(Frame_num)+"_Type1_standard.jpg",lip_rect)
      # plt.axis("off")
      # plt.imshow(lip_rect)
      # plt.show()

      lip_rect_small = cv2.resize(lip_rect_o, small_dimension)
      lip_rect_small = cv2.cvtColor(lip_rect_small, cv2.COLOR_BGR2RGB)
      cv2.imwrite(typeI_directory_small + "Frame"+str(Frame_num)+"_Type1_small.jpg",lip_rect_small)
      # plt.axis("off")
      # plt.imshow(lip_rect_small)
      # plt.show()
      return lip_rect, lip_rect_small



def typeII_2dFrame(image_rgb, landmark, typeII_directory_standard, typeII_directory_small, Frame_num):
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
    lip_mask : TYPE
        DESCRIPTION.
    lip_mask_small : TYPE
        DESCRIPTION.

    '''
    standard_dimension = (100,50) # first x then y 
    small_dimension = (50,25)
    pts_all = landmark[0,48:67,:]
    ## Crop the bounding rect
    rect = cv2.boundingRect(pts_all)
    x,y,w,h = rect
    croped = image_rgb[y:y+h, x:x+w].copy()
    ## make mask on the lip
    pts_all = pts_all - pts_all.min(axis=0)
    #
    mask = np.zeros(croped.shape[:2], np.uint8)
    ctr = np.array(pts_all).reshape((-1,1,2)).astype(np.int32)
    cv2.drawContours(mask, [ctr], -1, (255, 255, 255), -1, cv2.LINE_AA)
    lip_mask = cv2.resize(mask, standard_dimension)
    lip_mask = cv2.cvtColor(lip_mask, cv2.COLOR_BGR2RGB)
    cv2.imwrite(typeII_directory_standard + "Frame"+str(Frame_num)+"_Type2_standard.jpg",lip_mask)
    # plt.axis("off")
    # plt.imshow(lip_mask)
    # plt.show()

    lip_mask_small = cv2.resize(mask, small_dimension)
    lip_mask_small = cv2.cvtColor(lip_mask_small, cv2.COLOR_BGR2RGB)
    cv2.imwrite(typeII_directory_small + "Frame"+str(Frame_num)+"_Type2_small.jpg",lip_mask_small)
    # plt.axis("off")
    # plt.imshow(lip_mask_small)
    # plt.show()
    return lip_mask, lip_mask_small


def typeIII_2dFrame(image_rgb, landmark, typeII_directory_standard, typeII_directory_small,Frame_num):
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
    cv2.imwrite(typeII_directory_standard + "Frame"+str(Frame_num)+"_Type3_standard.jpg",lip_dst)
    # plt.axis("off")
    # plt.imshow(lip_dst)
    # plt.show()
    lip_dst_small = cv2.resize(dst, small_dimension)
    lip_dst_small = cv2.cvtColor(lip_dst_small, cv2.COLOR_BGR2RGB)
    cv2.imwrite(typeII_directory_small + "Frame"+str(Frame_num)+"_Type3_small.jpg",lip_dst_small)
    # plt.axis("off")
    # plt.imshow(lip_dst_small)
    # plt.show()
    return lip_dst, lip_dst_small


def typeIV_2dFrame(landmark, typeIV_directory,Frame_num):
    '''
    '''
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
    savemat(typeIV_directory+"Frame"+str(Frame_num)+"_Type4.mat",mdic)
    return dist_array


def rotation_once(landmark_array):
    landmark_list = list(landmark_array)
    landmark_list.append(landmark_list.pop(0))
    return np.array(landmark_list)

# @TODO: implement Type I, Type II, Type III, and Type IV 
# Type I 
# Type II 
# Type III 

    

#%%
######################################################################################
#                                       Test field                                   #
######################################################################################

# #%% FaceLandmark Test 
# Example_image = cv2.imread(Path_VideoImageExample_root+Path_fullFrames+'frame0.jpg')
# landmark = FacialLandmark(Example_image)

#%% itergration test
Video2Frames(Path_image_root+Video_name, Path_VideoImageExample_root+Path_fullFrames, Path_VideoImageExample_root+Path_2dFrames, Path_VideoImageExample_root+Path_3dMatrixs, Video_name)