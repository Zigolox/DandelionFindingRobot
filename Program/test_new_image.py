# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras


# Helper libraries
import glob, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
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

def test_classifier(model, image):
    prediction = model.predict(np.array([image/255]))
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image",image)
    y_pos = np.arange(len(class_names))

    plt.figure(figsize=(10,10))

    plt.bar(y_pos, prediction[0], align='center', alpha=0.5)
    plt.xticks(y_pos, class_names)
    plt.ylabel('Probability')
    plt.title('Classifier results')

    plt.show()


    cv2.waitKey(0)
    cv2.destroyAllWindows()

path = "../Images/" + input("File name: ")

img = np.array(cv2.imread(path, 1))/255
subList = createSubImages(img)

classifier = "dandelion_model_7.h5"#input("Which classifier will you use? ")
model = keras.models.load_model(classifier)

predictions = model.predict(np.array(subList))

#for im in subList:
#    test_classifier(model, im)

print(len(subList))
print("Test TEST")
a = 0

while(a+25 < len(subList)):
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(subList[i+a], cmap=plt.cm.binary)
        m = 0
        for j in range(len(predictions[i])):
            if(predictions[i+a][j] > predictions[i+a][m]):
                m = j
        plt.xlabel(class_names[m])
    plt.show()
    a += 25
