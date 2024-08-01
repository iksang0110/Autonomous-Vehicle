from __future__ import division
from torch.autograd import Variable
import torch.nn as nn
from darknet import Darknet, set_requires_grad
from shapely.geometry import Polygon, Point
from moviepy.editor import VideoFileClip
from PIL import Image
import numpy as np
import os
import io
import cv2
import argparse
import time
from math import *
from util import *

# Color definitions
red, green, blue = (0, 0, 255), (0, 255, 0), (255, 0, 0)
white, yellow, deepgray = (255, 255, 255), (0, 255, 255), (43, 43, 43)
dark, cyan, magenta = (1, 1, 1), (255, 255, 0), (255, 0, 255)
lime, purple = (0, 255, 128), (255, 0, 255)

font = cv2.FONT_HERSHEY_SIMPLEX
font2 = cv2.FONT_HERSHEY_PLAIN

# Global variable initialization
l_pos, r_pos, l_cent, r_cent = 0, 0, 0, 0
uxhalf, uyhalf, dxhalf, dyhalf = 0, 0, 0, 0
l_center, r_center, lane_center = ((0, 0)), ((0, 0)), ((0, 0))
next_frame = (0, 0, 0, 0, 0, 0, 0, 0)

# 프레임 버퍼 및 관련 변수 추가
frame_buffer = []
BUFFER_SIZE = 3
DISPLAY_INTERVAL = 1 / 30  # 30 FPS로 제한
STANDARD_SIZE = (1280, 720)

def arg_parse():
    parser = argparse.ArgumentParser(description='Test Autonomous Driving Vision System')
    parser.add_argument("--video", dest='video', default="/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/drive3.mp4", help="input video file")
    parser.add_argument("--roi", dest='roi', default=0, type=int, help="roi flag")
    parser.add_argument("--alpha", dest='alpha', default=0.2, type=float, help="alpha value for lane detection smoothing")
    return parser.parse_args()

args = arg_parse()
ALPHA = args.alpha 

def resize_frame(frame):
    return cv2.resize(frame, STANDARD_SIZE)

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
    global cache, first_frame, next_frame
    y_global_min = img.shape[0]
    y_max = img.shape[0]
    l_slope, r_slope = [], []
    l_lane, r_lane = [], []
    det_slope = 0.5

    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope = get_slope(x1,y1,x2,y2)
                if slope > det_slope:
                    r_slope.append(slope)
                    r_lane.append(line)
                elif slope < -det_slope:
                    l_slope.append(slope)
                    l_lane.append(line)
        y_global_min = min(y1, y2, y_global_min)

    if (len(l_lane) == 0 or len(r_lane) == 0):
        return 1

    l_slope_mean = np.mean(l_slope, axis=0)
    r_slope_mean = np.mean(r_slope, axis=0)
    l_mean = np.mean(np.array(l_lane), axis=0)
    r_mean = np.mean(np.array(r_lane), axis=0)

    if ((r_slope_mean == 0) or (l_slope_mean == 0)):
        print('dividing by zero')
        return 1

    l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0])
    r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])

    l_x1 = int((y_global_min - l_b)/l_slope_mean)
    l_x2 = int((y_max - l_b)/l_slope_mean)
    r_x1 = int((y_global_min - r_b)/r_slope_mean)
    r_x2 = int((y_max - r_b)/r_slope_mean)

    if l_x1 > r_x1:
        l_x1 = int((l_x1 + r_x1)/2)
        r_x1 = l_x1
        l_y1 = int((l_slope_mean * l_x1) + l_b)
        r_y1 = int((r_slope_mean * r_x1) + r_b)
        l_y2 = int((l_slope_mean * l_x2) + l_b)
        r_y2 = int((r_slope_mean * r_x2) + r_b)
    else:
        l_y1 = y_global_min
        l_y2 = y_max
        r_y1 = y_global_min
        r_y2 = y_max

    current_frame = np.array([l_x1, l_y1, l_x2, l_y2, r_x1, r_y1, r_x2, r_y2], dtype="int32")

    if first_frame == 1:
        next_frame = current_frame
        first_frame = 0
    else:
        prev_frame = cache
        next_frame = (1-ALPHA)*prev_frame + ALPHA*current_frame
        next_frame = next_frame.astype(int)

    if all(0 <= x < img.shape[1] for x in [next_frame[0], next_frame[2], next_frame[4], next_frame[6]]) and \
       all(0 <= y < img.shape[0] for y in [next_frame[1], next_frame[3], next_frame[5], next_frame[7]]):
        cv2.line(img, (next_frame[0], next_frame[1]), (next_frame[2], next_frame[3]), red, 2)
        cv2.line(img, (next_frame[4], next_frame[5]), (next_frame[6], next_frame[7]), red, 2)
    else:
        print("Invalid line coordinates")

    cache = next_frame

def lane_pts():
    pts = np.array([[next_frame[0], next_frame[1]], [next_frame[2], next_frame[3]], [next_frame[6], next_frame[7]], [next_frame[4], next_frame[5]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    return pts

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def get_pts(flag=0):
    vertices1 = np.array([[230, 650], [620, 460], [670, 460], [1050, 650]])
    vertices2 = np.array([[0, 720], [710, 400], [870, 400], [1280, 720]])
    return vertices1 if flag == 0 else vertices2

def direction_line(image, height, whalf, color=yellow):
    cv2.line(image, (whalf, height), (whalf, 600), white, 2)
    cv2.line(image, (whalf, height), (dxhalf, 600), red, 2)
    cv2.circle(image, (whalf, height), 120, white, 2)

def lane_position(image, gap=20, length=20, thickness=2, color=red, bcolor=white):
    global l_cent, r_cent
    l_left, l_right = 300, 520
    l_cent = int((l_left+l_right)/2)
    cv2.line(image, (l_center[0], l_center[1]+length), (l_center[0], l_center[1]-length), color, thickness)
    r_left, r_right = 730, 950
    r_cent = int((r_left+r_right)/2)
    cv2.line(image, (r_center[0], r_center[1]+length), (r_center[0], r_center[1]-length), color, thickness)

    def process_image(image):
        global first_frame
        image = resize_frame(image)
        height, width = image.shape[:2]

        gray_image = grayscale(image)
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_yellow = np.array([20, 100, 100], dtype="uint8")
        upper_yellow = np.array([30, 255, 255], dtype="uint8")
        mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(gray_image, 100, 255)
        mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
        mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)

        gauss_gray = gaussian_blur(mask_yw_image, 3)
        canny_edges = canny(gauss_gray, 150, 200)
        vertices = [get_pts(flag=0)]
        roi_image = region_of_interest(canny_edges, vertices)
        line_image = hough_lines(roi_image, 2, np.pi/180, 100, 50, 150)
        result = weighted_img(line_image, image, α=ALPHA, β=1., λ=0.)

        return result, line_image

def visualize(image, roi_flag):
    height, width = image.shape[:2]
    zeros = np.zeros_like(image)
    
    if roi_flag:
        vertices = get_pts(flag=roi_flag)
        cv2.polylines(zeros, [vertices], True, (0, 255, 255), 2)
    
    pts = lane_pts()
    if pts is not None:
        cv2.fillPoly(zeros, [pts], (0, 255, 0, 0.3))
    
    if l_center[0] != 0 and r_center[0] != 0:
        cv2.line(zeros, l_center, r_center, (0, 0, 255), 2)
    
    direction_line(zeros, height, width // 2)
    result = cv2.addWeighted(image, 1, zeros, 0.5, 0)
    return result

def write(x, results, color=[126, 232, 229], font_color=red):
    try:
        c1 = tuple(map(int, x[1:3]))
        c2 = tuple(map(int, x[3:5]))
        cls = int(x[-1])
        label = f"{classes[cls]}"
        cv2.rectangle(results, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(results, c1, c2, color, -1)
        cv2.putText(results, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    except Exception as e:
        print(f"Error in write function: {e}")
    return results

def process_image(image):
    global first_frame
    image = resize_frame(image)
    height, width = image.shape[:2]

    gray_image = grayscale(image)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20, 100, 100], dtype="uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 100, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)

    gauss_gray = gaussian_blur(mask_yw_image, 3)
    canny_edges = canny(gauss_gray, 150, 200)
    vertices = [get_pts(flag=0)]
    roi_image = region_of_interest(canny_edges, vertices)
    line_image = hough_lines(roi_image, 2, np.pi/180, 100, 50, 150)
    result = weighted_img(line_image, image, α=ALPHA, β=1., λ=0.)

    return result, line_image

def process_frame(frame):
    global frames
    
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
    frame = resize_frame(frame)
    
    cpframe = frame.copy()
    prc_img, hough = process_image(cpframe)
    lane_detection = visualize(prc_img, args.roi)

    if frames % 3 == 0:
        prep_frame = prep_image(frame, input_dim)
        frame_dim = frame.shape[1], frame.shape[0]
        frame_dim = torch.FloatTensor(frame_dim).repeat(1, 2)

        if CUDA:
            frame_dim = frame_dim.cuda()
            prep_frame = prep_frame.cuda()

        with torch.no_grad():
            output = model(Variable(prep_frame, True), CUDA)
        output = write_results(output, confidence, num_classes, nms_thesh)

        if type(output) is not int and output is not None and len(output) > 0:
            frame_dim = frame_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(resol/frame_dim, 1)[0].view(-1, 1)

            output[:, [1, 3]] -= (input_dim - scaling_factor * frame_dim[:, 0].view(-1, 1))/2
            output[:, [2, 4]] -= (input_dim - scaling_factor * frame_dim[:, 1].view(-1, 1))/2
            output[:, 1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, frame_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, frame_dim[i,1])

            zero_frame = np.zeros_like(frame)
            for det in output:
                if len(det) >= 7:
                    zero_frame = write(det, zero_frame)

            object_detection = cv2.add(frame, zero_frame)
            
            try:
                lane_detection = cv2.addWeighted(object_detection, 0.8, lane_detection, 0.5, 0)
            except cv2.error as e:
                print(f"Error in cv2.addWeighted: {e}")
                lane_detection = object_detection  # 오류 발생 시 object_detection만 사용
        else:
            lane_detection = cv2.addWeighted(frame, 0.8, lane_detection, 0.5, 0)

    return lane_detection

# Main code
print("[TEST] Initializing autonomous driving vision system...")

cfg = "/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/cfg/yolov3.cfg"
weights = "/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/cfg/yolov3.weights"
names = "/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/data/coco.names"

video = args.video

frames = 0
first_frame = 1

start = 0
batch_size = 1
confidence = 0.8
nms_thesh = 0.3
resol = 416

whalf, height = 640, 720

num_classes = 12
print("[TEST] Reading configuration files...")
model = Darknet(cfg)
print("[TEST] Loading weights...")
model.load_weights(weights)
print("[TEST] Loading classes...")
classes = load_classes(names)
set_requires_grad(model, False)
print("[TEST] Network successfully loaded!")

model.net_info["height"] = resol
input_dim = int(model.net_info["height"])
assert input_dim % 32 == 0
assert input_dim > 32

torch.cuda.empty_cache()

CUDA = torch.cuda.is_available()
if CUDA:
    model.cuda()
model.eval()

start = time.time()

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    print(f"Error: Could not open video file {video}")
    exit()

print("\n[TEST] Video is now ready to process.")

last_update_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream")
        break

    processed_frame = process_frame(frame)
    
    if processed_frame.shape[:2] == STANDARD_SIZE:
        frame_buffer.append(processed_frame)
        if len(frame_buffer) > BUFFER_SIZE:
            frame_buffer.pop(0)
    else:
        print(f"Skipping frame with non-standard size: {processed_frame.shape[:2]}")

    current_time = time.time()
    if current_time - last_update_time >= DISPLAY_INTERVAL:
        if frame_buffer:
            try:
                avg_frame = np.mean(frame_buffer, axis=0).astype(np.uint8)
                
                cv2.putText(avg_frame, f"FPS: {1/(current_time - last_update_time):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(avg_frame, f"ALPHA: {ALPHA:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow("Test - Autonomous Driving Vision", avg_frame)
            except ValueError as e:
                print(f"Error calculating average frame: {e}")
                if frame_buffer:
                    cv2.imshow("Test - Autonomous Driving Vision", frame_buffer[-1])
        
        last_update_time = current_time

    frames += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[TEST] Processing completed.")