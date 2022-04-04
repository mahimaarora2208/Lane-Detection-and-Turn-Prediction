'''
project 2 : problem 2

Straight Lane Detection
'''

import cv2 
import numpy as np

red = (0, 0, 255)
green = (0, 255, 0)
cap = cv2.VideoCapture('whiteline.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('lane_detection_video.avi',fourcc, 20.0, (960, 540))


def slope_of_lane(x1, y1, x2, y2):
    slope = (y2 - y1)/(x2 - x1)
    return slope

def length_of_lane(x1, y1, x2, y2):
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return length

while cap.isOpened():
    success, frame = cap.read()
    # frame = cv2.flip(q, flipCode = 1)
    # if not success: # when frame is not available, it loads the video again
    #     cap = cv2.VideoCapture('whiteline.mp4')
    #     continue
    if success:
        # mask img to get white lines
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS) # HSL used for color tracking
        lower = np.uint8([200, 200, 200])
        upper = np.uint8([255, 255, 255])
        white_mask = cv2.inRange(frame, lower, upper)
        masked = cv2.bitwise_and(frame, frame, mask = white_mask)
        # cv2.imshow('masked_img', masked)
        
        # Detect edges
        lane_edges = cv2.Canny(masked, 50, 150)
        # cv2.imshow('lane_edges', lane_edges)

        # Hough Lines
        lanes = cv2.HoughLinesP(lane_edges, 1, np.pi/160, 100, maxLineGap=50)
        if lanes is not None:
            for lane in lanes:
                x1, y1, x2, y2 = lane[0]
                slope = slope_of_lane(x1, y1, x2, y2)
                length = length_of_lane(x1, y1, x2, y2)
                if length > 170:
                    cv2.line(frame, (x1, y1),(x2, y2), green, 2) # green --> solid --> length is greater than a threshold 
                elif length < 170: 
                    cv2.line(frame, (x1, y1),(x2, y2), red, 2) # green --> dashed --> length is less than a threshold
                else:
                    print('length is invalid. Ignore points and Continue...')
                    continue      
            cv2.imshow('lanes_detected', frame)
            out.write(frame) 
            # print(frame.shape)
        key = cv2.waitKey(25)
        if key == 27:
           break
    else:
        break  
     
out.release()       
cap.release()
cv2.destroyAllWindows()

