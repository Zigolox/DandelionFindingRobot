# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras


# Helper libraries
import glob, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
TRAIN_IMG_SIZE = 200

def readFromCamera():
    #Read image from camera using opencv cv2
    pass

def writeToArduino():
    #Write action to arduino
    pass


def createSubImages(img):
    shape = (img.shape[0], img.shape[1]) #Shape of image
    N,M = shape[0]//TRAIN_IMG_SIZE, shape[1]//TRAIN_IMG_SIZE #amount of subimages
    subimgList = [] #List of all subimages
    for n in range(N):
        for m in range(M):
            subimg =img[n*TRAIN_IMG_SIZE:(n+1)*TRAIN_IMG_SIZE,m*TRAIN_IMG_SIZE:(m+1)*TRAIN_IMG_SIZE]
            subimg = cv2.resize(subimg,(50,50))
            subimgList.append(subimg)
    return subimgList

def findAll(subImages,model):
    out = [0]*len(subImages)
    predictions = model.predict(np.array(subImages))
    searchValues = [1,2,3,4]
    for i in range(len(predictions)):
        m = 0
        for j in range(len(predictions[i])):
            if(predictions[i][j] > predictions[i][m]):
                m = j
        if (m in searchValues):
            out[i] = 1

    return out

def Action(allData, W, H):
    #0:Straight Forward, 1: Right, 2: Left
    N = len(found)//H
    mid = N//2
    min_angle = 4
    min_pos = -1

    for i in range(allData):
        if(i == 1):
            if(abs(mid-i%H) < abs(mid-min_pos)):
                min_pos = i

    if((min_pos-mid) > min_angle ):
        return 1
    elif((mid-min_pos) > min_angle or min_pos == -1):
        return 2
    else:
        return 0
