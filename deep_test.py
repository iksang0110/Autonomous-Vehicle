import numpy as np
import cv2
from PIL import Image

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
white = (255, 255, 255)
yellow = (0, 255, 255)
deepgray = (43, 43, 43)

cyan = (255, 255, 0)
magenta = (255, 0, 255)
lime = (0, 255, 128)
font = cv2.FONT_HERSHEY_SIMPLEX

# Global 변수 초기화
l_center, r_center, lane_center = ((0,0)), ((0,0)), ((0,0))
pts = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.int32)
pts = pts.reshape((-1, 1, 2))

first_frame = 1

def grayscale(img):
    """Applies the Grayscale"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """Applies an image mask."""
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255, ) * channel_count
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

    y_global_min = img.shape[0] 
    y_max = img.shape[0]

    l_slope, r_slope = [], []
    l_lane, r_lane = [], []

    det_slope = 0.5
    α = 0.2

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

    l_slope_mean = np.mean(l_slope, axis =0)
    r_slope_mean = np.mean(r_slope, axis =0)
    l_mean = np.mean(np.array(l_lane), axis=0)
    r_mean = np.mean(np.array(r_lane), axis=0)

    if ((r_slope_mean == 0) or (l_slope_mean == 0 )):
        print('dividing by zero')
        return 1

    l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0])
    r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])

    if np.isnan((y_global_min - l_b)/l_slope_mean) or \
    np.isnan((y_max - l_b)/l_slope_mean) or \
    np.isnan((y_global_min - r_b)/r_slope_mean) or \
    np.isnan((y_max - r_b)/r_slope_mean):
        return 1

    l_x1 = int((y_global_min - l_b)/l_slope_mean)
    l_x2 = int((y_max - l_b)/l_slope_mean)
    r_x1 = int((y_global_min - r_b)/r_slope_mean)
    r_x2 = int((y_max - r_b)/r_slope_mean)

    if l_x1 > r_x1:
        l_x1 = int((l_x1 + r_x1)/2)
        r_x1 = l_x1

        l_y1 = int((l_slope_mean * l_x1 ) + l_b)
        r_y1 = int((r_slope_mean * r_x1 ) + r_b)
        l_y2 = int((l_slope_mean * l_x2 ) + l_b)
        r_y2 = int((r_slope_mean * r_x2 ) + r_b)

    else: 
        l_y1 = y_global_min
        l_y2 = y_max
        r_y1 = y_global_min
        r_y2 = y_max

    current_frame = np.array([l_x1, l_y1, l_x2, l_y2, r_x1, r_y1, r_x2, r_y2], dtype ="float32")

    if first_frame == 1:
        next_frame = current_frame
        first_frame = 0
    else:
        prev_frame = cache
        next_frame = (1-α)*prev_frame+α*current_frame

    global pts
    pts = np.array([[next_frame[0], next_frame[1]], [next_frame[2], next_frame[3]], [next_frame[6], next_frame[7]], [next_frame[4], next_frame[5]]], np.int32)
    pts = pts.reshape((-1, 1, 2))

    global l_center
    global r_center
    global lane_center

    div = 2
    l_center = (int((next_frame[0] + next_frame[2]) / div), int((next_frame[1] + next_frame[3]) / div))
    r_center = (int((next_frame[4] + next_frame[6]) / div), int((next_frame[5] + next_frame[7]) / div))
    lane_center = (int((l_center[0] + r_center[0]) / div), int((l_center[1] + r_center[1]) / div))

    cache = next_frame

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(image):
    global first_frame

    image = np.array(Image.fromarray(image).resize((1280, 720)))
    height, width = image.shape[:2]

    kernel_size = 3

    low_thresh = 100
    high_thresh = 150

    rho = 4
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

    vertices = [get_pts(image)]
    roi_image = region_of_interest(canny_edges, vertices)

    line_image = hough_lines(roi_image, rho, theta, thresh, min_line_len, max_line_gap)
    result = weighted_img(line_image, image, α=0.8, β=1., λ=0.)

    return result

def get_pts(image):
    height, width = image.shape[:2]

    vertices = np.array([
                [250, 650],
                [550, 470],
                [730, 470],
                [1100, 650]
                ])
    return vertices

def visualize(result):
    height, width = result.shape[:2]
    length = 30
    thickness = 3
    whalf = int(width/2)
    sl_color = yellow

    cv2.line(result, (whalf, lane_center[1]), (whalf, int(height)), sl_color, 2)
    cv2.line(result, (whalf, lane_center[1]), (lane_center[0], lane_center[1]), sl_color, 2)

    gap = 20
    legth2 = 10
    wb_color = white
    cv2.line(result, (whalf-gap, lane_center[1]-legth2), (whalf-gap, lane_center[1]+legth2), wb_color, 1)
    cv2.line(result, (whalf+gap, lane_center[1]-legth2), (whalf+gap, lane_center[1]+legth2), wb_color, 1)

    lp_color = red
    cv2.line(result, (l_center[0], l_center[1]), (l_center[0], l_center[1]-length), lp_color, thickness)
    cv2.line(result, (r_center[0], r_center[1]), (r_center[0], r_center[1]-length), lp_color, thickness)
    cv2.line(result, (lane_center[0], lane_center[1]), (lane_center[0], lane_center[1]-length), lp_color, thickness)

    hei = 30
    font_size = 2
    if lane_center[0] < whalf-gap:
        cv2.putText(result, 'WARNING : ', (10, hei), font, 1, red, font_size)
        cv2.putText(result, 'Turn Right', (190, hei), font, 1, red, font_size)
    elif lane_center[0] > whalf+gap:
        cv2.putText(result, 'WARNING : ', (10, hei), font, 1, red, font_size)
        cv2.putText(result, 'Turn Left', (190, hei), font, 1, red, font_size)

    return result

def Region(image):
    height, width = image.shape[:2]

    zeros = np.zeros_like(image)
    mask = cv2.fillPoly(zeros, [pts], lime)

    hhalf = int(height/2)
    if not lane_center[1] < hhalf:
        mask = visualize(mask)
    return mask

def Lane_Detection(image):
    processing = process_image(image)
    region = Region(processing)
    region_resized = cv2.resize(region, (image.shape[1], image.shape[0]))
    combined = cv2.addWeighted(image, 1, region_resized, 0.3, 0)
    return combined

#--------------------------Video test--------------------------------------

first_frame = 1

image_name = "/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/drive3.mp4"
cap = cv2.VideoCapture(image_name)

while (cap.isOpened()):
    _, frame = cap.read()
    if frame is None:
        break
    result = Lane_Detection(frame)

    cv2.imshow("result", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
