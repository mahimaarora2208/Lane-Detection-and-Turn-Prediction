'''
problem 3 : turn prediction
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
# defining color

red_color = (0, 0, 255)
yellow_color = (0, 255, 255)
blue_color = (255, 0, 0)

def measure_curvature(left_fit, right_fit):
    ym = 30/720
    xm = 3.7/700

    left_fit = left_fit.copy()
    right_fit = right_fit.copy()
    y_eval = 700 * ym

    # Compute R_curve (radius of curvature)
    left_curveR =  ((1 + (2*left_fit[0] *y_eval + left_fit[1])**2)**1.5)  / np.absolute(2*left_fit[0])
    right_curveR = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    xl = np.dot(left_fit, [700**2, 700, 1])
    xr = np.dot(right_fit, [700**2, 700, 1])
    pos = (1280//2 - (xl+xr)//2)*xm
    return left_curveR, right_curveR, pos



def get_lanes(img):
    hsl_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # Search for the yellow lane at left
    lower_mask_y = np.array([10, 0, 100], dtype='uint8')
    upper_mask_y = np.array([40, 255, 255], dtype='uint8')
    mask_yellow = cv2.inRange(hsl_img, lower_mask_y, upper_mask_y)
    yellow_lane = cv2.bitwise_and(hsl_img, hsl_img, mask=mask_yellow).astype(np.uint8)

    # Search for the white lane at right
    lower_mask_w = np.array([200, 200, 200], dtype='uint8')
    upper_mask_w = np.array([255, 255, 255], dtype='uint8')
    mask_white = cv2.inRange(hsl_img, lower_mask_w, upper_mask_w)
    white_lane = cv2.bitwise_and(hsl_img, hsl_img, mask=mask_white).astype(np.uint8)
    # Combine both for getting lanes
    yellow_white_lane = cv2.bitwise_or(yellow_lane, white_lane)
    img = cv2.cvtColor(yellow_white_lane, cv2.COLOR_HLS2BGR)
    cv2.imshow('masked(2)',img)
    return img


def lane_operations(warp):
    # Calculating the histogram to get max value
    hist = np.sum(warp, axis=0)
    img = np.dstack((warp, warp, warp))*255
    # cv2.imshow('im', img)
    mid_point = int(hist.shape[0] / 2)
    left_x_ind = np.argmax(hist[:mid_point])
    right_x_ind = np.argmax(hist[mid_point:]) + mid_point  # Add mid_point to avoid bias for 0

    img_center = int(warp.shape[1] / 2)
    # print(warp.shape)
    turn = turn_prediction(left_x_ind, right_x_ind, img_center)

    # Use the sliding window method to extract pixels
    num_window = 10
    # the margin of the width of the window
    margin = 50
    window_height = int(warp.shape[0] / num_window)
    # x, y positions of none zero pixels
    nonzero_pts = warp.nonzero()
    nonzero_x = np.array(nonzero_pts[1])
    nonzero_y = np.array(nonzero_pts[0])
    
    #current postion being updated for each window
    left_x_cur = left_x_ind
    right_x_cur = right_x_ind

    left_lane = []
    right_lane = []
    # Set min_num_pixels to recenter the window
    min_num_pixels = 50
    for i in range(num_window):
        # Set the window boundary in x and y, left and right
        window_y_low = warp.shape[0] - (i + 1) * window_height
        window_y_high = warp.shape[0] - i * window_height
        window_left_low = left_x_cur - margin
        window_left_high = left_x_cur + margin
        window_right_low = right_x_cur - margin
        window_right_high = right_x_cur + margin

        # Identify none zero pixels in the window
        left_lane_pixels = ((nonzero_y >= window_y_low)
                            & (nonzero_y < window_y_high)
                            & (nonzero_x >= window_left_low)
                            & (nonzero_x < window_left_high)).nonzero()[0]  # &: binary operator

        right_lane_pixels = ((nonzero_y >= window_y_low)
                             & (nonzero_y < window_y_high)
                             & (nonzero_x >= window_right_low)
                             & (nonzero_x < window_right_high)).nonzero()[0]

        left_lane.append(left_lane_pixels)
        right_lane.append(right_lane_pixels)

        # Shift to the next mean position if the length of none zero pixels is greater than min_num_pixels
        if len(left_lane) > min_num_pixels:
            left_x_cur = int(np.mean(nonzero_x[left_lane_pixels]))
        if len(right_lane) > min_num_pixels:
            right_x_cur = int(np.mean(nonzero_x[right_lane_pixels]))

    # Concatenate the arrays of left and right lane
    left_lane = np.concatenate(left_lane)
    right_lane = np.concatenate(right_lane)

    # Extract the left and right lane pixel positions
    left_x = nonzero_x[left_lane]
    left_y = nonzero_y[left_lane]
    right_x = nonzero_x[right_lane]
    right_y = nonzero_y[right_lane]

    img[nonzero_y[left_lane], nonzero_x[left_lane]] = yellow_color
    img[nonzero_y[right_lane], nonzero_x[right_lane]] = red_color
    # cv2.imshow('img', img)
    return img, left_x, left_y, right_x, right_y, turn


def poly_fit(img, left_x, left_y, right_x, right_y):
    # Use np.polyfit() to fit a second order polynomial
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)
    curvature = measure_curvature(left_fit, right_fit)
    
    print('=========================RADIUS OF CURVATURE =========================\n')
    print('left radius of curvature', curvature[0])
    print('right radius of curvature',  curvature[1], '\n')
    
    plot_y = np.linspace(0, img.shape[0] - 1, img.shape[0])

    left_fit_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
    right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]

    # Extract points from line fitting
    left_pts = np.array([(np.vstack([left_fit_x, plot_y])).T])
    # flip the array upside down in order to cv2.fillPoly()
    right_pts = np.array([np.flipud((np.vstack([right_fit_x, plot_y])).T)])

    pts = np.hstack((left_pts, right_pts))
    pts = np.array(pts, dtype='int32')

    img = np.zeros_like(img).astype(np.uint8)
    cv2.fillPoly(img, pts, red_color)
    cv2.polylines(img, np.int32([left_pts]), isClosed=False, color = blue_color, thickness=20)
    cv2.polylines(img, np.int32([right_pts]), isClosed=False, color=yellow_color, thickness=20)
    return img

def turn_prediction(left_lane_pts, right_lane_pts, image_center):
    center_lane = left_lane_pts + (right_lane_pts - left_lane_pts) / 2

    if abs(center_lane - image_center) < 10:
        return "Straight"

    elif center_lane - image_center < 0:
        return "Turn Right"

    else:
        return "Turn Left"

def main():
    file_dir = "./challenge.mp4"
    cap = cv2.VideoCapture(file_dir)
    # source points to be warped (points are determined through try and error)
    src_pts = np.float32([(550, 460),     # top-left
                            (150, 720),     # bottom-left
                            (1200, 720),    # bottom-right
                            (770, 460)])    # top-right
    dst_pts = np.float32([(100, 0),
                            (100, 720),
                            (1100, 720),
                            (1100, 0)])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('turn_prediction_video.avi',fourcc, 20.0, (1280, 720))
    if not cap.isOpened():
        print("Error")
    while cap.isOpened():
        success, frame = cap.read()        
        if success:
            height, width, _ = frame.shape
            extract_lane_img = get_lanes(frame)
            gray = cv2.cvtColor(extract_lane_img, cv2.COLOR_BGR2GRAY)
            # Filter noise
            img_blur = cv2.bilateralFilter(gray, 9, 120, 100)

            # Apply edge detection
            img_edge = cv2.Canny(img_blur, 100, 200)
            h, _ = cv2.findHomography(src_pts, dst_pts)
            warp = cv2.warpPerspective(img_edge, h, (width, height))
            cv2.imshow('warped(3)', warp)
            img, left_x, left_y, right_x, right_y, turn = lane_operations(warp)
           
            if np.sum(left_x) == 0 or np.sum(left_y) == 0 or np.sum(right_x) == 0 or np.sum(right_y) == 0:
                hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
                clahe_img = clahe.apply(hsv_img[:, :, 2])

                processed_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
                warp = cv2.warpPerspective(processed_img, h, (width, height))
                gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                img, left_x, left_y, right_x, right_y, turn = lane_operations(blur)
                lane_detect_imgx = poly_fit(img, left_x, left_y, right_x, right_y)
            else:
                lane_detect_img = poly_fit(img, left_x, left_y, right_x, right_y)

            # Unwarp the image
            h_inv = np.linalg.inv(h)
            lane_detect_img = cv2.warpPerspective(lane_detect_img, h_inv, (width, height))

            final_img = cv2.addWeighted(np.uint8(frame), 1, np.uint8(lane_detect_img), 0.5, 0)

            cv2.putText(final_img, turn, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, red_color, 2, cv2.LINE_AA)
            cv2.imshow('final output', final_img)
            out.write(final_img)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
