import numpy as np
import cv2
from PIL import Image
import math


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
font = cv2.FONT_HERSHEY_SIMPLEX

# Global 변수 초기화
l_center, r_center, lane_center = ((0,0)), ((0,0)), ((0,0))
pts = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.int32)
pts = pts.reshape((-1, 1, 2))

first_frame = 1
next_frame = np.zeros(8)
l_pos, r_pos, l_cent, r_cent = 0, 0, 0, 0
uxhalf, uyhalf, dxhalf, dyhalf = 0, 0, 0, 0

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

    div = 2
    l_center = (int((next_frame[0] + next_frame[2]) / div), int((next_frame[1] + next_frame[3]) / div))
    r_center = (int((next_frame[4] + next_frame[6]) / div), int((next_frame[5] + next_frame[7]) / div))
    lane_center = (int((l_center[0] + r_center[0]) / div), int((l_center[1] + r_center[1]) / div))

    uxhalf = int((next_frame[2]+next_frame[6])/2)
    uyhalf = int((next_frame[3]+next_frame[7])/2)
    dxhalf = int((next_frame[0]+next_frame[4])/2)
    dyhalf = int((next_frame[1]+next_frame[5])/2)

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

    # 이미지 크기 조정 (처리 속도 향상을 위해)
    image = cv2.resize(image, (960, 540))
    height, width = image.shape[:2]

    kernel_size = 5
    low_thresh = 50
    high_thresh = 150
    rho = 1  # 더 정밀한 라인 검출
    theta = np.pi/180
    thresh = 30  # 더 낮은 임계값으로 더 많은 선 검출
    min_line_len = 50  # 더 짧은 선분도 검출
    max_line_gap = 100  # 간격 증가로 끊어진 선 연결

    gray_image = grayscale(image)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # 노란색과 흰색 범위 확장
    lower_yellow = np.array([15, 80, 80], dtype="uint8")
    upper_yellow = np.array([35, 255, 255], dtype="uint8")
    lower_white = np.array([0, 0, 200], dtype="uint8")
    upper_white = np.array([255, 30, 255], dtype="uint8")

    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(img_hsv, lower_white, upper_white)

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
    # ROI 영역을 더 넓게 설정
    vertices = np.array([
                [int(0.1*width), height],
                [int(0.4*width), int(0.6*height)],
                [int(0.6*width), int(0.6*height)],
                [int(0.9*width), height]
                ])
    return vertices

def calculate_steering_angle(image, lines):
    height, width = image.shape[:2]
    
    if len(lines) == 0:
        return 0
    
    left_lines = []
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if slope < 0:
            left_lines.append(line)
        else:
            right_lines.append(line)
    
    left_points = np.array([l[0] for l in left_lines])
    right_points = np.array([l[0] for l in right_lines])
    
    # 왼쪽, 오른쪽 차선의 평균점 계산
    left_mean = np.mean(left_points, axis=0) if len(left_points) > 0 else None
    right_mean = np.mean(right_points, axis=0) if len(right_points) > 0 else None
    
    if left_mean is not None and right_mean is not None:
        center = (left_mean + right_mean) / 2
    elif left_mean is not None:
        center = left_mean
    elif right_mean is not None:
        center = right_mean
    else:
        return 0
    
    car_center = width / 2
    offset = center[0] - car_center
    
    # 조향각 계산 (단순화된 계산)
    steering_angle = math.degrees(math.atan(offset / height))
    return steering_angle

def draw_steering_direction(image, steering_angle):
    height, width = image.shape[:2]
    center_x = int(width / 2)
    center_y = height
    
    # 조향 방향 선 그리기
    length = 100
    angle_rad = math.radians(steering_angle)
    end_x = int(center_x - length * math.sin(angle_rad))
    end_y = int(center_y - length * math.cos(angle_rad))
    
    cv2.line(image, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3)
    cv2.putText(image, f"Angle: {steering_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def Lane_Detection(image):
    processing = process_image(image)
    
    # 차선 검출
    gray = cv2.cvtColor(processing, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=100)
    
    # 조향각 계산
    steering_angle = calculate_steering_angle(processing, lines)
    
    region = visualize(processing)
    region_resized = cv2.resize(region, (image.shape[1], image.shape[0]))
    combined = cv2.addWeighted(image, 1, region_resized, 0.3, 0)
    
    # 조향 방향 표시
    draw_steering_direction(combined, steering_angle)
    
    # 색상 감지 및 표시
    color, point = detect_color(image)
    if color != "None":
        cv2.circle(combined, point, 10, (0, 255, 255), -1)  # 감지 지점 표시
        cv2.putText(combined, f"Detected: {color}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    return combined


def get_pts(image):
    height, width = image.shape[:2]
    
    # 사다리꼴 모양의 ROI 정의
    bottom_left = [int(0.1 * width), height]
    bottom_right = [int(0.9 * width), height]
    top_left = [int(0.4 * width), int(0.6 * height)]
    top_right = [int(0.6 * width), int(0.6 * height)]
    
    vertices = np.array([bottom_left, top_left, top_right, bottom_right], np.int32)
    
    return [vertices]


def warning_text(image):
    global dxhalf
    whalf, height = 640, 720
    center = whalf - 5
    angle = int(round(math.atan((dxhalf-center)/120) * 180/math.pi, 3) * 3)

    m = 2
    limit = 0
    if angle > 90:
        angle = 89
    if 90 > angle > limit:
        cv2.putText(image, 'WARNING : ', (10, 30*m), font, 0.8, red, 1)
        cv2.putText(image, 'Turn Right', (150, 30*m), font, 0.8, red, 1)

    if angle < -90:
        angle = -89
    if -90 < angle < -limit:
        cv2.putText(image, 'WARNING : ', (10, 30*m), font, 0.8, red, 1)
        cv2.putText(image, 'Turn Left', (150, 30*m), font, 0.8, red, 1)

    elif angle == 0:
        cv2.putText(image, 'WARNING : ', (10, 30*m), font, 0.8, white, 1)
        cv2.putText(image, 'None', (150, 30*m), font, 0.8, white, 1)

def direction_line(image, height, whalf, color=yellow):
    cv2.line(image, (whalf-5, height), (whalf-5, 600), white, 2)  # 방향 제어 기준선
    cv2.line(image, (whalf-5, height), (dxhalf, 600), red, 2)  # 핸들 방향 제어
    cv2.circle(image, (whalf-5, height), 120, white, 2)

def lane_position(image, gap=20, length=20, thickness=2, color=red, bcolor=white):
    global l_cent, r_cent

    l_left = 300
    l_right = 520
    l_cent = int((l_left+l_right)/2)
    cv2.line(image, (l_center[0], l_center[1]+length), (l_center[0], l_center[1]-length), color, thickness)

    r_left = 730
    r_right = 950
    r_cent = int((r_left+r_right)/2)
    cv2.line(image, (r_center[0], r_center[1]+length), (r_center[0], r_center[1]-length), color, thickness)

def draw_lanes(image, thickness=3, color=red):
    cv2.line(image, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]), int(next_frame[3])), red, 3)
    cv2.line(image, (int(next_frame[6]), int(next_frame[7])), (int(next_frame[4]), int(next_frame[5])), red, 3)

def visualize(result):
    height, width = result.shape[:2]
    whalf = int(width/2)
    hhalf = int(height/2)

    zeros = np.zeros_like(result)
    
    if not lane_center[1] < hhalf:
        cv2.fillPoly(zeros, [pts], lime)
        lane_position(zeros)
        direction_line(zeros, height=height, whalf=whalf)
        draw_lanes(zeros)
        warning_text(zeros)

    return zeros

def Lane_Detection(image):
    processing = process_image(image)
    region = visualize(processing)
    region_resized = cv2.resize(region, (image.shape[1], image.shape[0]))
    combined = cv2.addWeighted(image, 1, region_resized, 0.3, 0)
    return combined

def detect_color(image):
    height, width = image.shape[:2]
    y = int(height * 2/5)  # 화면 위에서 5분의 2 지점
    x = int(width / 2)  # 화면 가로 중앙

    # ROI 설정 (detection point 주변의 작은 영역)
    roi = image[y-10:y+10, x-10:x+10]
    
    # BGR에서 HSV로 변환
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 초록색 범위 정의
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    
    # 빨간색 범위 정의 (HSV에서 빨간색은 0-10 또는 170-180)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # 마스크 생성
    mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)
    mask_red1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # 초록색과 빨간색 픽셀 수 계산
    green_pixels = cv2.countNonZero(mask_green)
    red_pixels = cv2.countNonZero(mask_red)
    
    # 감지된 색상 결정
    if green_pixels > red_pixels and green_pixels > 20:  # 임계값 설정
        return "Green", (x, y)
    elif red_pixels > green_pixels and red_pixels > 20:  # 임계값 설정
        return "Red", (x, y)
    else:
        return "None", (x, y)

def Lane_Detection(image):
    processing = process_image(image)
    region = visualize(processing)
    region_resized = cv2.resize(region, (image.shape[1], image.shape[0]))
    combined = cv2.addWeighted(image, 1, region_resized, 0.3, 0)
    
    # 색상 감지 및 표시
    color, point = detect_color(image)
    if color != "None":
        cv2.circle(combined, point, 10, (0, 255, 255), -1)  # 감지 지점 표시
        cv2.putText(combined, f"Detected: {color}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    return combined
#--------------------------Video test--------------------------------------

first_frame = 1
cache = np.zeros(1)

image_name = "/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/drive3.mp4"
cap = cv2.VideoCapture(image_name)

frame_count = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    # 3프레임마다 한 번씩 처리
    if frame_count % 5 == 0:
        result = Lane_Detection(frame)
        # 결과를 원본 크기로 다시 확대
        result = cv2.resize(result, (1280, 720))
        cv2.imshow("result", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
'''
if __name__ == "__main__":
    first_frame = 1
    cache = np.zeros(1)

    cap = cv2.VideoCapture(3)  # 외부 카메라 사용

    # 카메라 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("외부 카메라를 열 수 없습니다. 내장 카메라를 사용합니다.")
        cap = cv2.VideoCapture(0)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 받아올 수 없습니다.")
            break
        
        frame_count += 1
        # 2프레임마다 한 번씩 처리 (15fps로 처리)
        if frame_count % 2 == 0:
            result = Lane_Detection(frame)
            result = cv2.resize(result, (1920, 1080))  # 원본 크기로 복원
            cv2.imshow("Lane Detection", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
'''