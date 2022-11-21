# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 10:17:37 2021

@author: postulate-2
"""

import pandas as pd
import os
import cv2


# tf=pd.read_csv("PCBData/group00041/00041_not/00041000.txt",index_col=None,sep=" ",header=None)
# img=cv2.imread("PCBData/group00041/00041/00041000_test.jpg")
# k=0
# name="00041000"
# for i in range(0,tf.shape[0]):
#     x=tf.iloc[i,0]
#     y=tf.iloc[i,1]
#     x2=tf.iloc[i,2]
#     y2=tf.iloc[i,3]
#     target=tf.iloc[i,4]
#     print(x,y,x2,y2,target)
#     cropped_image = img[y:y2, x:x2]
#     cropped_image=cv2.resize(cropped_image,(30,30))
#     cv2.imwrite(f"output/{target}/{name}_k.jpg",cropped_image)
#     cv2.rectangle(img, (x, y), (x2, y2), (255,150,0), 2)
#     k+1
# cv2.imshow("uytu",img)
    

group_list=["00041","12000","12100","12300","13000","20085","44000","50600","77000","90100","92000"]

for gp in group_list:    
    path=gp
    path1=f"PCBData/group{path}/{path}"
    list1=list(os.listdir(f"PCBData/group{path}/{path}"))
    
    im_c=len(list1)
    start=""
    
    for im in list1:
        c1=im.split(".")[0]
        c2=c1[-4:]
        if c2=="temp":
            continue
        path2=f"{path1}/{im}"
        tn=c1[:-5]
        tf=pd.read_csv(f"PCBData/group{path}/{path}_not/{tn}.txt",index_col=None,sep=" ",header=None)
        img=cv2.imread(path2)
        k=0
        name=tn
        for i in range(0,tf.shape[0]):
            x=tf.iloc[i,0]
            y=tf.iloc[i,1]
            x2=tf.iloc[i,2]
            y2=tf.iloc[i,3]
            target=tf.iloc[i,4]
    #        print(x,y,x2,y2,target)
            cropped_image = img[y:y2, x:x2]
            cropped_image=cv2.resize(cropped_image,(30,30))
            cv2.imwrite(f"output/{target}/{name}_{k}.jpg",cropped_image)
    #        cv2.rectangle(img, (x, y), (x2, y2), (255,150,0), 2)
            k+=1
    #    cv2.imshow("uytu",img)
        
                        
