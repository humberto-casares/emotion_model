import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import os
import scipy.misc
import dlib
import cv2
from PIL import Image
import pickle

from PIL import Image
from skimage.feature import hog

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from keras.utils import to_categorical
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, concatenate, Input
from keras.layers import Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, Adagrad
from keras.regularizers import l2
from keras.initializers import he_normal
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

from keras.utils.vis_utils import plot_model
from IPython.display import Image, display



window_size = 24
window_step = 6
height = 48
width = 48


predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def get_landmarks(image, rects):
    # this function have been copied from http://bit.ly/2cj7Fpq
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])


def sliding_hog_windows(image):
    hog_windows = []
    for y in range(0, height, window_step):
        for x in range(0, width, window_step):
            window = image[y:y+window_size, x:x+window_size]
            hog_windows.extend(hog(window, orientations=8, pixels_per_cell=(8, 8),
                                            cells_per_block=(1, 1), visualise=False))
    return hog_windows


# get images and extract features
images = []
labels = []
#landmarks = []
hog_features = []
               
for emotion in os.listdir("../data2/"):
    path = os.listdir(os.path.join("../data2", emotion))
    length = len(path)
    for (i, image) in enumerate(path):
        print(str(i) + ' / ' + str(length) + ' processed')
        
        img = cv2.imread(os.path.join("../data2/" + emotion, image))
        img = cv2.resize(img, (48, 48))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        images.append(np.array(gray, dtype='float32').reshape((48, 48, 1)))
        
        features = sliding_hog_windows(gray)
       # f, hog_image = hog(gray, orientations=8, pixels_per_cell=(16, 16),
        #                cells_per_block=(1, 1), visualise=True)
        hog_features.append(features)
        
        #face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
        #face_landmarks = get_landmarks(img, face_rects)
        #landmarks.append(face_landmarks) 
        labels.append(emotion)
        
images = np.asarray(images, dtype="float32")
images /= 255
#images = images.astype('float32')

le = LabelEncoder()
labels = np.array(labels)
labels = le.fit_transform(labels)
labels = to_categorical(labels, num_classes=3)

#landmarks = np.array(landmarks, dtype='float32')
#landmarks = np.array([x.flatten() for x in landmarks])
hog_features = np.array(hog_features, dtype='float32')
#landmarks = np.concatenate([landmarks, hog_features], axis=1)
#landmarks /= landmarks.max()

np.save('images.npy', images)
np.save('labels.npy', labels)
#np.save('landmarks.npy', landmarks)
np.save('hog_features.npy', hog_features)

images = np.load("images.npy")
labels = np.load("labels.npy")
hog_features = np.load("hog_features.npy")

BS = 128
EPOCHS = 50
lr = 0.01
num_classes = 3

#construct CNN structure
inputA = Input(shape=(48, 48, 1))
x = Conv2D(64, (3, 3), kernel_initializer=he_normal())(inputA)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(x)

#2nd convolution layer
x = Conv2D(128, (3, 3), kernel_initializer=he_normal())(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(x)


#3rd convolution layer
x = Conv2D(256, (3, 3), kernel_initializer=he_normal())(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(x)

#x = Dropout(0.5)(x)
x = Flatten()(x)

#fully connected neural networks
#model.add(Dense(1024, activation='relu'))
#model.add(Dropout(0.5))
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Model(inputs=inputA, outputs=x)

inputB = Input(shape=(2728, ))
x2 = Dense(1024, activation='relu')(inputB)
x2 = Dense(128, activation='relu')(x2)
x2 = Model(inputs=inputB, outputs=x2)

combined = concatenate([x.output, x2.output])
combined = Dense(num_classes, activation='softmax')(combined)

model = Model([x.input, x2.input], outputs=combined)
model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

plot_model(model, 'model.png', show_shapes=True, show_layer_names=True)
display(Image(filename="model.png"))