import numpy as np
import matplotlib.pyplot as plt
import cv2
TRAIN_IMG_SIZE = 200

class_names = [
"other",
"blooming dandelion",
"seeding dandelion",
"cleared dandelion",
"unbloomed dandelion",
"buttercup",
"grass",
"cow parsley",
"other flower",
"road",
]

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


    #TEST:
##    print(len(subimgList))
##    for i in range(30):
##        cv2.namedWindow('image'+str(i),cv2.WINDOW_NORMAL)
##        cv2.imshow('image'+str(i), subimgList[10*i])
##    print(len(subimgList))
##    cv2.waitKey(0)  #Wait for keyboard input
##    cv2.destroyAllWindows()   #Close all windows

def classify(img, impath = "./", labpath = "./", file_name = "test"):
    print("Possible classes:")
    for i in range(len(class_names)):
        print(str(i)+": "+class_names[i])
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test', np.array(img))
    label = cv2.waitKey(0)  #Wait for keyboard input, this is apperently neccesary
    print("Which class is this (integer value): ")
    with open(labpath+file_name+".txt","w") as f:
        f.write(chr(label))
    cv2.imwrite(impath+file_name+".png", img)
    cv2.destroyAllWindows()


cv2.destroyAllWindows()

first = 46
last = 62

for i in range(first,last+1):
    if(i<10):
        path = "../train_data/Unclassified_Images/IMG_0"+str(i)+".JPG"
    else:
        path = "../train_data/Unclassified_Images/IMG_"+str(i)+".JPG"
    img = np.array(cv2.imread(path, 1))
    subList = createSubImages(img)

    for j in range(len(subList)):
        im_path = "../train_data/Features/"
        lab_path = "../train_data/Labels/"
        name = "data_"+str(i)+"_"+str(j)

        classify(subList[j], im_path, lab_path, name)
