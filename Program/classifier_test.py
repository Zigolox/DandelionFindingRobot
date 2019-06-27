from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras


# Helper libraries
import glob, os
import numpy as np
import cv2
import matplotlib.pyplot as plt

class_names = [
"other",
"bloom dand",
"seed dand",
"clear dand",
"unbloom dand",
"buttercup",
"grass",
"cow pars",
"other flower",
"road",
]

def test_classifier(model, image, label):
    prediction = model.predict(np.array([image/255]))
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image",image)
    y_pos = np.arange(len(class_names))

    plt.figure(figsize=(10,10))

    plt.bar(y_pos, prediction[0], align='center', alpha=0.5)
    plt.xticks(y_pos, class_names)
    plt.ylabel('Probability')
    plt.title('Classifier results')


    cv2.waitKey(0)
    cv2.destroyAllWindows()

def importData(feat_path, lab_path):
    img = cv2.imread(feat_path,1)
    with open(lab_path,"r") as f:
        label = int(f.read())
    return img, label

feature_path = "../train_data/Features/"
label_path = "../train_data/Labels/"

classifier = "dandelion_model_5.h5"#input("Which classifier will you use? ")
model = keras.models.load_model(classifier)

#name = input("Which file would you like to see? ")
#feature, label = importData(feature_path+name+".png",label_path+name+".txt")
#test_classifier(model, feature, label)

with open("test_files.txt","r") as f:
    files = f.read().split(".png")
for name in files:
    print(name)
    feature, label = importData(feature_path+name+".png",label_path+name+".txt")
    test_classifier(model, feature, label)
#for file_name in os.listdir("./train_data/Features/"):
#    if(file_name == ".DS_Store"):
#        continue
#    name = file_name.replace(".txt","")
#    print(name)
#    feature, label = importData(feature_path+name+".png",label_path+name+".txt")
#    test_classifier(model, feature, label)
