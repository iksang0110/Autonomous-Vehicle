import cv2
import numpy as np
import time
from math import *

# Color definitions (기존과 동일)
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
white = (255, 255, 255)
yellow = (0, 255, 255)
deepgray = (43, 43, 43)
dark = (1, 1, 1)
cyan = (255, 255, 0)
magenta = (255, 0, 255)
lime = (0, 255, 128)
purple = (255, 0, 255)

font = cv2.FONT_HERSHEY_SIMPLEX
font2 = cv2.FONT_HERSHEY_PLAIN

# Global variables
l_pos, r_pos, l_cent, r_cent = 0, 0, 0, 0
uxhalf, uyhalf, dxhalf, dyhalf = 0, 0, 0, 0
l_center, r_center, lane_center = ((0, 0)), ((0, 0)), ((0, 0))
next_frame = (0, 0, 0, 0, 0, 0, 0, 0)
first_frame = 1
# Global variables (기존과 동일)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def get_slope(x1,y1,x2,y2):
    return (y2-y1)/(x2-x1)

def draw_lines(img, lines):
    global cache
    global first_frame
    global next_frame
    global l_center, r_center, lane_center
    global uxhalf, uyhalf, dxhalf, dyhalf

    # (기존 코드와 동일)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def get_pts(flag=0):
    # (기존 코드와 동일)
    vertices1 = np.array([
                [230, 650],
                [620, 460],
                [670, 460],
                [1050, 650]
                ])
    vertices2 = np.array([
                [0, 720],
                [710, 400],
                [870, 400],
                [1280, 720]
    ])
    return vertices1 if flag == 0 else vertices2

def process_image(image):
    global first_frame

    height, width = image.shape[:2]

    kernel_size = 3
    low_thresh = 150
    high_thresh = 200
    rho = 2
    theta = np.pi/180
    thresh = 100
    min_line_len = 50
    max_line_gap = 150

    gray_image = grayscale(image)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower_yellow = np.array([20, 100, 100], dtype = "uint8")
    upper_yellow = np.array([30, 255, 255], dtype = "uint8")

    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 100, 255)

    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)

    gauss_gray = gaussian_blur(mask_yw_image, kernel_size)

    canny_edges = canny(gauss_gray, low_thresh, high_thresh)

    vertices = [get_pts(flag = 0)]
    roi_image = region_of_interest(canny_edges, vertices)

    line_image = hough_lines(roi_image, rho, theta, thresh, min_line_len, max_line_gap)
    result = weighted_img(line_image, image, α=0.8, β=1., λ=0.)

    return result, line_image

def visualize(image):
    height, width = image.shape[:2]
    whalf = int(width/2)
    hhalf = int(height/2)

    zeros = np.zeros_like(image)
    pts = np.array([[next_frame[0], next_frame[1]], [next_frame[2], next_frame[3]], 
                    [next_frame[6], next_frame[7]], [next_frame[4], next_frame[5]]], np.int32)
    pts = pts.reshape((-1, 1, 2))

    if not lane_center[1] < hhalf:
        if r_center[0]-l_center[0] > 100:
            cv2.fillPoly(zeros, [pts], lime)
            cv2.line(zeros, (l_center[0], l_center[1]+20), (l_center[0], l_center[1]-20), red, 2)
            cv2.line(zeros, (r_center[0], r_center[1]+20), (r_center[0], r_center[1]-20), red, 2)
            cv2.line(zeros, (whalf-5, height), (whalf-5, 600), white, 2)
            cv2.line(zeros, (whalf-5, height), (dxhalf, 600), red, 2)
            cv2.circle(zeros, (whalf-5, height), 120, white, 2)

    return zeros

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    print("\n[INFO] Video is now ready to show.")
    
    frames = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream")
            break

        prc_img, hough = process_image(frame)
        lane_detection = visualize(prc_img)

        result = cv2.addWeighted(frame, 1, lane_detection, 0.5, 0)

        cv2.putText(result, f"FPS: {frames / (time.time() - start):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, yellow, 2)

        cv2.imshow("Lane Detection", result)
        frames += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/drive3.mp4"
    try:
        process_video(video_path)
    except Exception as e:
        print(f"An error occurred: {e}")