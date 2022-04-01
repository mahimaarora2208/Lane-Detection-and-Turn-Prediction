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


import cv2
import numpy as np
from matplotlib import pyplot as plt

def histogram_img(img):
    pass

def cumulative_sum():
    pass


# img = cv2.imread('data_1/0000000000.png',0)
img = cv2.imread('clahe_2.jpg',0)
hist,bins = np.histogram(img.flatten(),256,[0,256])
 
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[img]
# cv2.imwrite('clahe_2.jpg',img2)
# def adjust_gamma(image, gamma=1.0):
#     # build a lookup table mapping the pixel values [0, 255] to
#     # their adjusted gamma values
#     invGamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** invGamma) * 255
#                       for i in np.arange(0, 256)]).astype("uint8")
#     # apply gamma correction using the lookup table
#     return cv2.LUT(image, table)


# def main():
 
#     # cap = cv2.VideoCapture('./Night Drive - 2689.mp4')
#     a = np.zeros((256,),dtype=np.float16)
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter('Night_Drive_improved.avi', fourcc, 20.0, (1920, 1080))
#     # if not cap.isOpened():
#         # print("Error")
#     # while cap.isOpened():
#         # ret, frame = cap.read()
#         if ret:
#             height, width, _ = frame.shape
#             blur = cv2.GaussianBlur(frame, (7, 7), 0)
#             # Convert the color image to HSV, in order to apply histogram equalization method to the "V" channel
#             hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
#             # Instantiate the CLAHE algorithm using cv2.createCLAHE()
#             clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
#             # Call the .apply method on the CLAHE object to apply histogram equalization
#             clahe_img = clahe.apply(hsv[:, :, 2])
            
#             gamma_img = adjust_gamma(clahe_img, 1.0)
#             hsv[:, :, 2] = gamma_img

#             processed_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


#             # showing the image
#             cv2.imshow('improved_image', processed_img)
#             out.write(processed_img)
#             if cv2.waitKey(30) & 0xFF == ord("q"):
#                 break
#         else:
#             break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     main()