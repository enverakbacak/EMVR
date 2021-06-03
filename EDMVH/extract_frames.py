
#listing = os.listdir("/home/ubuntu/Desktop/EMR/Article_Video/Video_Dataset_3/videoFolder/")

#In addition to v3, also creates a *.npy file for each video (video's frames), and save them to respective data's subfolders
import os
import math
import cv2
import re
import matplotlib.pyplot as plt
import numpy as np

listing = os.listdir("/home/ubuntu/Desktop/EMR/Article_Video/Video_Dataset_3/videoFolder/")
count = 1
for file in listing:
    cap = cv2.VideoCapture("/home/ubuntu/Desktop/EMR/Article_Video/Video_Dataset_3/videoFolder/" + file)
    os.makedirs("/home/ubuntu/Desktop/EMR/Article_Video/Video_Dataset_3/Frames/" + file + "/")
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    def getFrame(sec):
        cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = cap.read()
        if hasFrames:
            e=cv2.imwrite("/home/ubuntu/Desktop/EMR/Article_Video/Video_Dataset_3/Frames/"+ file + "/" + file + "_" +str(count)+".jpg", image)
        return hasFrames
    sec = 0
    N = 33 # How many frames will be extracted
    interwall = duration/N   #//it will capture image in each 'interwall' second
    count = 1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + interwall
        success = getFrame(sec)
    os.system('sh /home/ubuntu/keras/enver/edmvh/resize.sh')
'''
    X = []
    frame_dir = os.listdir("/home/ubuntu/Desktop/EMR/Article_Video/Video_Dataset_3/Frames/" + file  + "/")
    for im in frame_dir:
        frame =  plt.imread("/home/ubuntu/Desktop/EMR/Article_Video/Video_Dataset_3/Frames/" + file + "/" + im)
        X.append (frame)
    X = np.array(X)
    np.save(open("/home/ubuntu/Desktop/EMR/Article_Video/Video_Dataset_3/NP_videos3/" + file  + ".npy", 'w'), X)
'''



