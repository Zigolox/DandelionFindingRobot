from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

from tensorflow.keras.layers import Dropout, Flatten, Activation, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

# Helper libraries
import glob, os
import numpy as np
import cv2
import matplotlib.pyplot as plt

#class_names = [
#"other",
#"blooming dandelion",
#"seeding dandelion",
#"cleared dandelion",
#"unbloomed dandelion",
#"buttercup",
#"grass",
#"dogwood",
#"other flower",
#"road",
#]
class_names = [
"other",
"dandelion",
"buttercup"
]
class_amounts = [0]*len(class_names)

def yellowPixel(pix): #Check if a pixel is yellow
    b, g, r = pix[0], pix[1], pix[2]
    val = b < min(g,r)/2
    val = val and min(g,r) > 130
    val = val and max(g,r) < 1.3*min(g,r)

    return val

def countYellow(img):
    N = 0
    for row in img:
        for pixel in row:
            if(yellowPixel(pixel)):
                N += 1
    return N

def sortData(Data, amounts, focus):
    features, labels = Data
    i = 0
    while i < len(labels):
        if(amounts[labels[i]] > 1.5*amounts[focus]):
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
    k1 = 0
    k2 = 0
    k3 = 0
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
                        if(i > 9 or i < 0):
                            continue
                    except ValueError:
                        print("Error: invalid data in file")
                        print("File: "+lab_path + file_name.replace(".JPG",".txt").replace(".png",".txt"))
                        print("Value: " + i)
                        continue

                if(i == 1):
                    lab = 1
                    k1 += 1
                elif(i == 5):
                    lab = 2
                    k2 += 2
                else:
                    lab = 0
                    k3 += 1

                if(k1%10 == 0):
                    test_f.write(file_name)
                    test_img.append(img)
                    test_labels.append(lab)
                    k1+=1
                elif(k2%10 == 0):
                    test_f.write(file_name)
                    test_img.append(img)
                    test_labels.append(lab)
                    k2+=1
                elif(k3%10 == 0):
                    test_f.write(file_name)
                    test_img.append(img)
                    test_labels.append(lab)
                    k3+=1

                else:
                    train_img.append(img)
                    train_labels.append(lab)
                    class_amounts[lab] += 1

            except FileNotFoundError:
                print("Error: .txt file not found")
                print("Path: "+lab_path + file_name.replace(".JPG",".txt").replace(".png",".txt"))
                continue


    train_img, train_labels = sortData((train_img,train_labels), class_amounts, 1)
    print(class_amounts)
    return np.array(train_img), np.array(train_labels), np.array(test_img), np.array(test_labels)

feature_path = "../large_train_data/Features/"
label_path = "../large_train_data/Labels/"

allData = readData(feature_path,label_path)

train_images, train_labels, test_images, test_labels = allData

print("Shape of train data:", train_images.shape)
print("Shape of test data:", test_images.shape)





model = keras.Sequential([
    Conv2D(64,(2, 2),
    padding='same',
    input_shape = (256,256,3),
    activation = tf.nn.relu,
    kernel_regularizer=regularizers.l2(2e-8),
    bias_regularizer=regularizers.l2(2e-8)) ,

    Conv2D(64,(4,4),
    padding='same',
    activation = tf.nn.relu,
    kernel_regularizer=regularizers.l2(2e-8),
    bias_regularizer=regularizers.l2(2e-8)),

    MaxPooling2D(pool_size=(3, 3)),

    Dropout(0.25),

    Conv2D(32,(2, 2),
    padding='same',
    activation = tf.nn.relu,
    kernel_regularizer=regularizers.l2(2e-8),
    bias_regularizer=regularizers.l2(2e-8)),

    Conv2D(32,(4,4),
    padding='same',
    activation = tf.nn.relu,
    kernel_regularizer=regularizers.l2(2e-8),
    bias_regularizer=regularizers.l2(2e-8)),

    MaxPooling2D(pool_size=(3, 3)),

    Dropout(0.25),

    Flatten(),

    Dense(128, activation = tf.nn.relu,
                kernel_initializer='random_uniform',
                bias_initializer='random_uniform',
                kernel_regularizer=regularizers.l2(2e-8),
                activity_regularizer=regularizers.l2(2e-8)),

    Dropout(0.25),

    Dense(128, activation = tf.nn.relu,
                kernel_initializer='random_uniform',
                bias_initializer='random_uniform',
                kernel_regularizer=regularizers.l2(2e-8),
                activity_regularizer=regularizers.l2(2e-8)),
    
    Dropout(0.25),


    Dense(10,activation = tf.nn.softmax,
                kernel_initializer='random_uniform',
                bias_initializer='random_uniform',)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

lossSetTest = []
lossSetTrain = []
accSetTest = []
accSetTrain = []
iSet=[]
for i in range(0,200):
    train_loss, train_acc = model.fit(train_images, train_labels, epochs = 1)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    lossSetTest.append(test_loss)
    lossSetTrain.append(train_loss)
    accSetTest.append(test_acc)
    accSetTrain.append(train_acc)
    iSet.append(i)
    model.save('dandelion_model_1.h5')
    print("Test loss:",test_loss)
    print("Train loss:",train_loss)
    print("Epoch:",i+1)

plt.figure(figsize=(10,10))
plt.plot(iSet,accSetTest,"r-",label ="Test")
plt.plot(iSet,accSetTrain,"b-",label = "Train")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

plt.figure(figsize=(10,10))
plt.plot(iSet,lossSetTest,"r-",label ="Test")
plt.plot(iSet,lossSetTrain,"b-",label = "Train")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

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
