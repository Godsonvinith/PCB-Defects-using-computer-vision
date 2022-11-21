# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 18:32:42 2021

@author: Roshini
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
import easygui
from skimage.feature import match_template

import os, numpy, PIL
from PIL import Image
from tkinter import filedialog as fd

# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim

#pip install scikit-learn==0.16.1
#import argparse
import imutils

from skimage.measure import label,regionprops, regionprops_table
from skimage import data, filters, measure, morphology

from skimage import measure

import pandas as pd

from tkinter.messagebox import showinfo
import tkinter as tk
from tkinter import *
import tensorflow as tf

root = Tk()
root.withdraw()


model = tf.keras.models.load_model('best_model_bpn.h5')

    
def load_image(img):
    # img=cv2.imread("output/3/00041009_5.jpg",0)
    threshold = filters.threshold_otsu(img)
    mask = img > threshold
    mask = morphology.remove_small_objects(mask, 50)
    mask = morphology.remove_small_holes(mask, 50)
    labels = measure.label(mask)
    label_img = label(labels)

    regions = regionprops(label_img)
    
    l=[]
    k=0
    for props in regions:
        l=[]
        l.append(props.filled_area)
        l.append(props.convex_area)
        l.append(props.eccentricity)
        l.append(props.equivalent_diameter)
        l.append(props.extent)
        l.append(props.perimeter)
        l.append(props.perimeter_crofton)
        l.append(props.solidity)
        l.append(props.area)
        
    
        print("Filled Area:",l[0])
        print("Convex Area:",l[1])
        print("Eccentricity:",l[2])
        print("Equivalent Diameter:",l[3])
        print("Extent:",l[4])
        print("Perimeter:",l[5])
        print("Perimeter Crofton:",l[6])
        print("Solidity:",l[7])
        print("Area:",l[8])
        return l



color=[(0,100,255),(255,255,0),(255,0,255),(0,0,255),(205,100,0),(0,255,0)]

color_text=[(0,100,255),(255,255,0),(255,0,255),(0,0,255),(205,100,0),(0,255,0)]
def Bpn_classifier(img_rgb):    
    X=[]
    #img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_rgb=cv2.resize(img_rgb,(30,30))
    dt=load_image(img_rgb)
    if dt!= None:
        dfg=np.array(dt)
        dfg=dfg.reshape(1,9)
        X.append(dfg)
        xd=np.array(X)
        xd=xd.reshape(xd.shape[0],9)
        dt=pd.DataFrame(xd)
        
        cl=['Open',
         'Short',
         'Mousebite',
         'Spur',
         'Copper',
         'Pin-hole']
        p1=model.predict(dt)
        
        p2=list(p1[0])
        p3=max(p2)
        p4=p2.index(p3)
        
        p5=cl[p4]
        return p5,p4
    return "None",7

#ij = np.unravel_index(np.argmax(result), result.shape)
#x, y = ij[::-1]
    
showinfo(
        title='alert',
        message=f"ready to selct original image"
    )
root.update()
original_image_path=fd.askopenfilename()
imageA = cv2.imread(original_image_path)
s=imageA.shape
imageA=cv2.resize(imageA,(500,500))
showinfo(
        title='alert',
        message=f"ready to selct Second image image"
    )

root.update()
edited=fd.askopenfilename()

root.quit()
root.destroy()

imageB = cv2.imread(edited)
imageB=cv2.resize(imageB,(500,500))
# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")

# print("SSIM: {}".format(score))


# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 150, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
            

# loop over the contours
i=0
imgc=imageB.copy()
res_all=[]
for c in cnts:
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
    (x, y, w, h) = cv2.boundingRect(c)
    #cv2.rectangle(imageB, (x, y,z), (x + w, y + h), (0, 0, 255), 2)
    cropped_image = grayB[y:y+h, x:x+w]
    res,index=Bpn_classifier(cropped_image)
   
    res_all.append(res)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if index !=7:
        print("Defect Number:",i+1)
        print("width:",w)
        print("Height:",h)
        print("Area Of Defect:",w*h)
        print("Defect:",res)
        cv2.rectangle(imgc, (x, y), (x + w, y + h), color[index], 2)
        cv2.putText(imgc, f'{i+1}.{res}', (x,y), font, 0.5,color_text[index] , 1, cv2.LINE_AA)
    
    print("\n")

    i+=1

cv2.imwrite(f"plot/result.jpg",imgc)
# show the output images
#print("Total Defect:",res_all)

print("Total Number Of Defect:",len(res_all))
print("\n")
print("list of Defects:")
k=1
for i in res_all:
    print(f"{k}.{i}")
    k=k+1

    

cv2.imshow("original", imageA)
cv2.imshow("Defected Location", imgc)
#imageB=cv2.resize(imageB,(500,500))
cv2.imshow("modified", imageB)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)#press q for close all window
cv2.destroyAllWindows()

 