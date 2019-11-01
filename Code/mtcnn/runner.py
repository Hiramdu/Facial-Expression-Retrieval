import tensorflow as tf

import keras
from keras.ac_ac_models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import sys

#------------------------------
#cpu - gpu configuration
config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 8} ) #max: 1 gpu, 56 cpu
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

#------------------------------
#variables
batch_size = 256
epochs = 5


def analyse_action(imagePath):
    #------------------------------
    ac_classes = 5 
    #construct CNN structure
    ac_model = Sequential()

    #1st convolution layer
    ac_ac_model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
    ac_model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

    #2nd convolution layer
    ac_model.add(Conv2D(64, (3, 3), activation='relu'))
    ac_model.add(Conv2D(64, (3, 3), activation='relu'))
    ac_model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    #3rd convolution layer
    ac_model.add(Conv2D(128, (3, 3), activation='relu'))
    ac_model.add(Conv2D(128, (3, 3), activation='relu'))
    ac_model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    ac_model.add(Flatten())

    #fully connected neural networks
    ac_model.add(Dense(1024, activation='relu'))
    ac_model.add(Dropout(0.2))
    ac_model.add(Dense(1024, activation='relu'))
    ac_model.add(Dropout(0.2))

    ac_model.add(Dense(ac_classes, activation='softmax'))

    #------------------------------

    ac_model.compile(loss='categorical_crossentropy'
        , optimizer=keras.optimizers.Adam()
        , metrics=['accuracy']
    )

    #------------------------------

    ac_model.load_weights('/home/hshuo/action_model_weights.h5') #load weights
    
    #------------------------------
    #make prediction for custom image out of test set
    img = image.load_img(imagePath, grayscale=True, target_size=(48, 48))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)

    x /= 255
    
    pred = ac_model.predict(x)
    return np.argsort(-pred)[:2] 
    #------------------------------

def analyse_emotion(imagePath):
    #------------------------------
    em_classes = 7
    #construct CNN structure
    em_model = Sequential()

    #1st convolution layer
    em_model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
    em_model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

    #2nd convolution layer
    em_model.add(Conv2D(64, (3, 3), activation='relu'))
    em_model.add(Conv2D(64, (3, 3), activation='relu'))
    em_model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    #3rd convolution layer
    em_model.add(Conv2D(128, (3, 3), activation='relu'))
    em_model.add(Conv2D(128, (3, 3), activation='relu'))
    em_model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    em_model.add(Flatten())

    #fully connected neural networks
    em_model.add(Dense(1024, activation='relu'))
    em_model.add(Dropout(0.2))
    em_model.add(Dense(1024, activation='relu'))
    em_model.add(Dropout(0.2))

    em_model.add(Dense(em_classes, activation='softmax'))

    #------------------------------

    em_model.compile(loss='categorical_crossentropy'
        , optimizer=keras.optimizers.Adam()
        , metrics=['accuracy']
    )

    #------------------------------

    em_model.load_weights('/home/hshuo/facial_expression_model_weights.h5') #load weights
    
    #------------------------------
    #make prediction for custom image out of test set
    img = image.load_img(imagePath, grayscale=True, target_size=(48, 48))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)

    x /= 255
    
    pred = em_model.predict(x)
    return np.argsort(-pred)[:2] 
    #------------------------------
    
    
if __name__ == "__main__":
    imgPath = sys.argv[1]
    action  = analyse_action(imgPath)
    emotion = analyse_emotion(imagePath)
    actionList = ['natural', 'up', 'down', 'left', 'right']
    emotionList = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    with open("/home/hshuo/res.txt",'w') as f:
        f.write(" ".join(actionList[action])+"\t"+" ".join(emotionList[emotion]))    