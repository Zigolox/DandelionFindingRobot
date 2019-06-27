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
"blooming dandelion",
"seeding dandelion",
"cleared dandelion",
"unbloomed dandelion",
"buttercup",
"grass",
"dogwood",
"other flower",
"road",
]
class_amounts = [0]*len(class_names)

def sortData(Data, amounts, focus):
    features, labels = Data
    i = 0
    while i < len(labels):
        if(amounts[labels[i]] > 5*amounts[focus]):
            labels.pop(i)
            features.pop(i)
            class_amounts[labels[i]]-=1
        else:
            i+=1
        if(i%100 == 0):
            i+=1
    return features,labels





def readData(feat_path = "../test_feat/", lab_path = "../test_lab/"):
    train_img = []
    train_labels = []
    test_img = []
    test_labels = []
    k = 0
    error = False
    grass_count = 0
    with open("test_files.txt","w+") as test_f:
        for file_name in os.listdir(feat_path)[1:]:
            if(file_name == ".DS_Store"):
                continue
            try:
                img = np.array(cv2.imread(feat_path+file_name))
            except FileNotFoundError:
                print("Error: Image file not found")
                print("Path: "+path+file_name)
                continue
            try:
                img = img/255
            except ValueError:
                print("Error: array contains non numbers")
                continue
            try:
                with open(lab_path + file_name.replace(".JPG",".txt").replace(".png",".txt") , "r") as f:
                    i = f.read()
                    try:
                        i = int(i)
                        if(i>9):
                            continue
                    except ValueError:
                        print("Error: invalid data in file")
                        print("File: "+lab_path + file_name.replace(".JPG",".txt").replace(".png",".txt"))
                        print("Value: " + i)
                        continue

                lab = i

                if(k%20 == 0):
                    test_f.write(file_name)
                    test_img.append(img)
                    test_labels.append(lab)

                else:
                    train_img.append(img)
                    train_labels.append(lab)
                    class_amounts[lab] += 1

            except FileNotFoundError:
                print("Error: .txt file not found")
                print("Path: "+lab_path + file_name.replace(".JPG",".txt").replace(".png",".txt"))
                continue


            k+=1
    train_img, train_labels = sortData((train_img,train_labels), class_amounts, 1)
    print(class_amounts)
    return np.array(train_img), np.array(train_labels), np.array(test_img), np.array(test_labels)

feature_path = "../train_data/Features/"
label_path = "../train_data/Labels/"

allData = readData(feature_path,label_path)

train_images, train_labels, test_images, test_labels = allData

print("Shape of train data:", train_images.shape)
print("Shape of test data:", test_images.shape)



model = keras.Sequential([
    keras.layers.Flatten(input_shape=(50,50,3)),
    keras.layers.Dense(64, activation = tf.nn.relu,
                kernel_initializer='random_uniform',
                bias_initializer='random_uniform'),
    keras.layers.Dense(10,activation = tf.nn.softmax,
                kernel_initializer='random_uniform',
                bias_initializer='random_uniform')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs = 30)
model.save('dandelion_model_5.h5')
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:',test_acc)

predictions = model.predict(test_images)
plt.figure(figsize=(10,10))
for i in range(36):
    plt.subplot(6,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i*3], cmap=plt.cm.binary)
    m = 0
    for j in range(len(predictions[i*3])):
        if(predictions[i*3][j] > predictions[i*3][m]):
            m = j
    plt.xlabel(class_names[m])
plt.show()
