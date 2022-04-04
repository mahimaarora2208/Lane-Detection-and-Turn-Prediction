'''
project2 : problem 1 
The aim is to improve the quality of the image sequence(25 frames).This
is a video recording of a highway during night. Most of the Computer Vision pipelines
for lane detection or other self-driving tasks require good lighting conditions and color
information for detecting good features. A lot of pre-processing is required in such
scenarios where lighting conditions are poor.
Now, using the techniques of computer vision, the goal is to enhance the contrast and
improve the visual appearance of the video sequence. 
I will use histogram equalization based methods, both
histogram equalization and adaptive histogram equalization and compare the results.

@author : Mahima Arora
'''

#!/usr/bin/env python


import time
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


def cumulative_sum(hist):
    hist = iter(hist)
    b = [next(hist)]
    for x in hist:
        b.append(b[-1] + x)
    return np.array(b)


def histogram_equalization(img, bins):
    img_arr = np.asarray(img)
    flat = img_arr.flatten()
    hist = np.zeros(bins)
    # loop through each pixel to count intensity
    for pixel in flat:
        hist[pixel] += 1 
    hist = cumulative_sum(hist) 
    hist = hist / hist[-1] # to convert values between 0 to 1
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img[x, y] = 255 * hist[img[x, y]]
    return img


def adaptive_equalization(img, tilesize):
    new_img = np.zeros_like(img)
    for x in range(0, img.shape[0], tilesize): 
        for y in range(0, img.shape[1], tilesize):
            new_img[x : x + tilesize, y : y + tilesize] = histogram_equalization(img[x : x + tilesize, y : y + tilesize], 256)
    return new_img



def main():
    file_dir = "./data_1/" # contains image frames
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('histogram_equalization_video.avi',fourcc, 20.0, (1224, 370), isColor = False)
    out1 = cv2.VideoWriter('adaptive_histogram_equalization_video.avi',fourcc, 20.0, (1224, 370), isColor = False)
    data = [x for x in os.listdir(file_dir) if x.endswith('.png')]
    # for sorting the file names properly
    data.sort()
    for file in data:
        filename = file_dir + file
        frame = cv2.imread(filename)
        cv2.imshow('input frame', frame)
        gray_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        histogram_eq = histogram_equalization(gray_img, 256)
        cv2.imshow('Histogram Equalisation ', histogram_eq)
        # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
        # clahe_img_1 = clahe.apply(gray_img)
        # converted_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        # cv2.imwrite('image_3.png', dst)
        adap_histogram_eq = adaptive_equalization(gray_img, 9)
        cv2.imshow('Adaptive Histogram Equalisation', adap_histogram_eq)
        out.write(histogram_eq)
        out1.write(adap_histogram_eq)
        # print(histogram_eq.shape)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    else:
        exit()  

    out.release()
    time.sleep(2)
    cv2.destroyAllWindows()
        # hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # ycrcb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        # hsv_planes = cv2.split(ycrcb_img)
        # hsv_planes[0] = histogram_equalization(hsv_planes[0], 256)
        # hsv_planes[1] = histogram_equalization(hsv_planes[1], 256)
        # hsv_planes[2] = histogram_equalization(hsv_planes[2], 256)
        # hsv_merge = cv2.merge(hsv_planes)
        # histogram_eq = cv2.cvtColor(hsv_merge, cv2.COLOR_HSV2BGR)
        # new_ycrcb_img = cv2.cvtColor(hsv_merge, cv2.COLOR_YCR_CB2BGR)
        # histogram_eq = histogram_equalization(hsv_img, 256)

      
    
if __name__ == '__main__':
    main()
