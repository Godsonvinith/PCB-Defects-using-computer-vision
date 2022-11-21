# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 17:20:40 2022

@author: postulate-41
"""

from skimage.measure import label,regionprops, regionprops_table
from skimage import data, filters, measure, morphology
import pandas as pd
import cv2
from skimage import measure
import  matplotlib.pyplot as plt
import os
import numpy as np


import keras
import tensorflow as tf

    
    
def load_image(img):
    # img=cv2.imread("output/3/00041009_5.jpg",0)
    threshold = filters.threshold_otsu(img)
    mask = img > threshold
    mask = morphology.remove_small_objects(mask, 50)
    mask = morphology.remove_small_holes(mask, 50)
    labels = measure.label(mask)
    label_img = label(labels)

    regions = regionprops(label_img)
    k=0
    l=[]
    for props in regions:
        # print('\n filled area',props.filled_area)
        # # print('filled image',props.filled_image)
        # # print('images ',props.image)
        # # print('bounding',props.bbox)
        # print('convex_area',props.convex_area)
        # # print('coords',props.coords)
        # print('eccentricity',props.eccentricity)
        # print('equivalent_diameter',props.equivalent_diameter)
        # print('euler_number',props.euler_number)
        # print('extent',props.extent)
        # # print('local_centroid',props.local_centroid)
        # print('perimeter',props.perimeter)
        # print('perimeter_crofton',props.perimeter_crofton)
        # # print('slice',props.slice)
        # print('solidity',props.solidity)
        # print('area',props.area)
        l.append(props.filled_area)
        l.append(props.convex_area)
        l.append(props.eccentricity)
        l.append(props.equivalent_diameter)
        l.append(props.extent)
        l.append(props.perimeter)
        l.append(props.perimeter_crofton)
        l.append(props.solidity)
        l.append(props.area)
        return l
    

ori_path=os.listdir('output/')
N=len(os.listdir('output'))

X=[]
Y=[]
cl=[]

for i in range(N):
#    print(i)
    cl.append(ori_path[i])
    path1=f"output/{ori_path[i]}"
    
    list1=os.listdir(path1)
    k=0
    for j in list1:
        path2=f"{path1}/{j}"
        image=cv2.imread(path2,0)
        dt=load_image(image)
        if dt!= None:
            dfg=np.array(dt)
            dfg=dfg.reshape(1,9)    
            X.append(dfg)
            Y.append(i)
            
        k=k+1

xd=np.array(X)
yd=np.array(Y)
xd=xd.reshape(xd.shape[0],9)
yd=tf.keras.utils.to_categorical(Y)
dt=pd.DataFrame(xd)


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(dt,yd,test_size=0.03,shuffle=True,random_state=True)



model = keras.Sequential([
    keras.layers.Dense(32,input_dim=9, activation='relu'),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=256, activation='relu'),
    keras.layers.Dense(units=512, activation='relu'),
    keras.layers.Dense(units=1024, activation='relu'),
    keras.layers.Dense(units=6, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# history = model.fit(
#     xtrain,ytrain,
#     epochs=10, 
#     steps_per_epoch=500,
#     validation_data=(xtest,ytest), 
#     validation_steps=2
# )


my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy',
                          min_delta = 0,
                          patience = 10,
                          verbose = 1,
                          restore_best_weights = True),
    tf.keras.callbacks.ModelCheckpoint(filepath='godson1.h5',monitor='val_accuracy',
                             mode='max',
                             save_best_only=True,
                             verbose=1),
]


model.fit(xtrain, ytrain, epochs=50, steps_per_epoch=500,
                    validation_data=(xtest, ytest),callbacks=my_callbacks)