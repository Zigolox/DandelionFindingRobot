# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras


# Helper libraries
import glob, os
import numpy as np
import cv2
from math import sin, cos, tan, pi, sqrt, asin

import serial
from time import sleep

SPLIT_IMG_SIZE = 20
PIXLE_ANGLE_CONV = 0.28
CAMERA_HEIGHT = 200
CAMERA_ANGLE = 0.87
CAMERA_POSITION = 40

def createSubImages(img):
    shape = (img.shape[0], img.shape[1]) #Shape of image
    N, M = shape[0]//SPLIT_IMG_SIZE, shape[1]//SPLIT_IMG_SIZE #amount of subimages
    print("N,M:", N, M)
    subimgList = [] #List of all subimages
    iCoordinates = []
    jCoordinates = []
    for n in range(N):
        for m in range(M):
            subimg =img[n*SPLIT_IMG_SIZE:(n+1)*SPLIT_IMG_SIZE,m*SPLIT_IMG_SIZE:(m+1)*SPLIT_IMG_SIZE]
            subimgList.append(subimg)
            iCoordinates.append((n*N + SPLIT_IMG_SIZE//2)-N//2)
            jCoordinates.append(M//2-(m*M + SPLIT_IMG_SIZE//2))
    return subimgList, iCoordinates, jCoordinates

def yellowPixel(pix): #Check if a pixel is yellow
    h, s, v = pix[0], pix[1], pix[2]
    #lowYellow = np.array([20,50,55])
    #highYellow = np.array([30,255,255])
    val = h < 30 and h > 20 and s > 50 and v > 55
    return val

def countYellow(img):
    N = 0
    i = 0
    lowYellow = np.array([22,51,57])
    highYellow=np.array([30,200,200])
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowYellow, highYellow)

    N = cv2.countNonZero(mask)
    #for row in img:
    #    for pixel in row:
    #        if(yellowPixel(pixel)):
    #            N += 1
    #        i += 1

    return N


def findAll(img):
    subImages, iC, jC = createSubImages(img)
    print("Subimages created")
    smallest = 0

    out = []
    n = 0
    m = 0

    for i in range(len(subImages)):
        if(countYellow(subImages[i]) > SPLIT_IMG_SIZE**2//3):
            out.append((iC[i],jC[i]))
            if(abs(iC[i] < m or n == 0)):
                smallest = n
                m = iC[i]
            n += 1
    print("All yellow found")

    return out, smallest

def pixToAngle(i,j,conv): #Calculate angles to located object
    x = i
    y = j

    alpha = tan(conv*x)
    beta = tan(conv*y)

    return alpha, beta

def calculateDistance(h,alpha, alpha0):
    dis = h*tan(alpha+alpha0)
    return dis

def calcCurveRadius(dis, beta):
    #Calculate radius of curvature from distance to dandelion and angle from heading
    if(abs(beta) < pi/20):
        R = 0
        a = 0
    elif(beta > 0):
        R = abs(dis*cos(beta)/sin(2*beta))
        a = 1
    else:
        R = abs(dis*cos(beta)/sin(2*beta))
        a = 2
    return R,a

def realDistance(dis, beta, L):
    disTrue = sqrt(dis**2+L**2+2*dis*L*cos(beta))
    betaTrue = asin(dis/disTrue*sin(beta))

    return disTrue,betaTrue


def sendData(R,a):
    ser.write((str(int(R))+ "."+str(a)+"\n").encode())
    sleep(1)


ser = serial.Serial("/dev/ttyACM0", 9600)
sleep(3)

R = calcCurveRadius(1000*sqrt(5), 0.463647609)

a = 1

cap = cv2.VideoCapture(0)


#if cap.isOpened():
    #window_handle = cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)
print("Starting")

while(True):
    ret, img = cap.read()
    #img = cv2.flip(img, 1)
    #cv2.imshow('CSI Camera',img)
    # This also acts as 
    #keyCode = cv2.waitKey(1) & 0xff
    # Stop the program on the ESC key
    #if keyCode == 27:
    #    break

    print("Image Read")
    yellow, smallest = findAll(img)
    print("Smallest pos:", yellow[smallest][0], yellow[smallest][1])
    alpha, beta = pixToAngle(yellow[smallest][0],yellow[smallest][1],PIXLE_ANGLE_CONV)
    D = calculateDistance(CAMERA_HEIGHT,alpha, CAMERA_ANGLE)
    print("Distance before:", D)
    print("Angle before:", beta)

    D, beta = realDistance(D, beta, CAMERA_POSITION)
    print("Distance after:", D)
    print("Angle after:", beta)
    R, a = calcCurveRadius(D, beta)
    print("R, a", R, a)
    sendData(R,a)
    print("Data sent")


    #line = ser.readline()
    #print("Line read")
    #line = line.decode("ascii") #ser.readline returns a binary, convert to string
    #print("Incoming line:",line)

    

cap.release()
cv2.destroyAllWindows()

