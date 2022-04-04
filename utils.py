import cv2
from matplotlib import pyplot as plt
import numpy as np
import glob

def frame_to_video():
    img_array = []
    for filename in glob.glob('/home/mahima/spring_2022/ENPM673/project2/data_1/*.png'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    
    out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def hist_frame():
    img = cv2.imread('data_1/0000000007.png',0)
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    plt.plot(cdf_normalized, color = 'y')
    plt.hist(img.flatten(),256,[0,256], color = 'b')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper right')
    plt.show()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[img]
    cv2.imshow('hist_new', img2)
    cv2.waitKey(0)

hist_frame()    