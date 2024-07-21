import numpy as np
import cv2
from PIL import Image
import math

# 색상 정의
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

# 전역 변수 초기화
l_center, r_center, lane_center = ((0,0)), ((0,0)), ((0,0))
pts = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.int32)
pts = pts.reshape((-1, 1, 2))

first_frame = 1
next_frame = np.zeros(8)
l_pos, r_pos, l_cent, r_cent = 0, 0, 0, 0
uxhalf, uyhalf, dxhalf, dyhalf = 0, 0, 0, 0
cache = np.zeros(8)
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
    return (y2-y1)/(x2-x1) if x2 != x1 else 0

def get_pts(image):
    height, width = image.shape[:2]
    bottom_left = (0, height)
    bottom_right = (width, height)
    bottom_center = (width // 2, height)
    apex_left = (width // 2 - width // 6, height // 2)
    apex_right = (width // 2 + width // 6, height // 2)

    vertices = np.array([
        [bottom_left, apex_left, apex_right, bottom_right]
    ], dtype=np.int32)

    return vertices
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

    if (len(l_lane) == 0 or len(r_lane) == 0):  # 차선이 감지되지 않은 경우
        return 1

    l_slope_mean = np.mean(l_slope, axis=0)
    r_slope_mean = np.mean(r_slope, axis=0)
    l_mean = np.mean(np.array(l_lane), axis=0)
    r_mean = np.mean(np.array(r_lane), axis=0)

    if ((r_slope_mean == 0) or (l_slope_mean == 0)):
        print('dividing by zero')
        return 1

    # 차선의 시작점과 끝점 계산
    l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0])
    r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])

    l_x1 = int((y_global_min - l_b)/l_slope_mean)
    l_x2 = int((y_max - l_b)/l_slope_mean)
    r_x1 = int((y_global_min - r_b)/r_slope_mean)
    r_x2 = int((y_max - r_b)/r_slope_mean)

    if l_x1 > r_x1:  # 좌우 차선이 교차하는 경우 보정
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
    pts = np.array([[next_frame[0], next_frame[1]], [next_frame[2], next_frame[3]], 
                    [next_frame[6], next_frame[7]], [next_frame[4], next_frame[5]]], np.int32)
    pts = pts.reshape((-1, 1, 2))

    div = 2
    l_center = (int((next_frame[0] + next_frame[2]) / div), int((next_frame[1] + next_frame[3]) / div))
    r_center = (int((next_frame[4] + next_frame[6]) / div), int((next_frame[5] + next_frame[7]) / div))
    lane_center = (int((l_center[0] + r_center[0]) / div), int((l_center[1] + r_center[1]) / div))

    uxhalf = int((next_frame[2]+next_frame[6])/2)
    uyhalf = int((next_frame[3]+next_frame[7])/2)
    dxhalf = int((next_frame[0]+next_frame[4])/2)
    dyhalf = int((next_frame[1]+next_frame[5])/2)

    cv2.line(img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]), int(next_frame[3])), (0, 0, 255), 2)
    cv2.line(img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), (0, 0, 255), 2)

    cache = next_frame
def process_image(image):
    height, width = image.shape[:2]

    # 이미지 전처리 강화
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 밝기 및 대비 조정
    gray = cv2.equalizeHist(gray)
    
    # 노이즈 제거
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 엣지 검출
    edges = cv2.Canny(gray, 50, 150)
    
    # ROI 마스크 적용
    vertices = get_pts(image)
    roi_mask = np.zeros_like(edges)
    cv2.fillPoly(roi_mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, roi_mask)
    
    # 차선 검출
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=100)
    
    # 차선 그리기
    line_image = np.zeros((height, width, 3), dtype=np.uint8)
    draw_lines(line_image, lines)
    
    # 원본 이미지와 차선 이미지 합성
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    
    return result, masked_edges

def calculate_steering_angle(image, lines):
    height, width = image.shape[:2]
    
    if lines is None:
        return 0
    
    left_lines = []
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.5:
            continue
        if slope < 0:
            left_lines.append(line)
        else:
            right_lines.append(line)
    
    left_points = np.array([l[0] for l in left_lines])
    right_points = np.array([l[0] for l in right_lines])
    
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
    
    steering_angle = math.degrees(math.atan(offset / height))
    return steering_angle

def draw_steering_direction(image, steering_angle):
    height, width = image.shape[:2]
    center_x = int(width / 2)
    center_y = height
    
    length = 100
    angle_rad = math.radians(steering_angle)
    end_x = int(center_x - length * math.sin(angle_rad))
    end_y = int(center_y - length * math.cos(angle_rad))
    
    cv2.line(image, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3)
    cv2.putText(image, f"Angle: {steering_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def detect_corner(steering_angle):
    threshold = 10  # 코너 감지를 위한 임계값
    if abs(steering_angle) > threshold:
        if steering_angle > 0:
            return "Right Corner"
        else:
            return "Left Corner"
    return "Straight"
def Lane_Detection(image):
    # 이미지 처리 및 차선 검출
    result, edges = process_image(image)
    
    # ROI 영역 표시
    vertices = get_pts(image)
    cv2.polylines(result, vertices, isClosed=True, color=(255, 0, 0), thickness=2)
    
    # 차선 중앙 지점 표시
    if lane_center != ((0,0)):
        cv2.circle(result, lane_center, 5, red, -1)
    
    # 조향각 계산 및 시각화
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(gray, 1, np.pi/180, 50, minLineLength=50, maxLineGap=100)
    steering_angle = calculate_steering_angle(result, lines)
    draw_steering_direction(result, steering_angle)
    
    # 코너 감지 및 표시
    corner_status = detect_corner(steering_angle)
    cv2.putText(result, corner_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 6)
    cv2.putText(result, corner_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return result, edges

if __name__ == "__main__":
    video_path = "/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/drive3.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("비디오 파일을 열 수 없습니다.")
        exit()

    # 비디오 저장을 위한 설정 (선택사항)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("비디오의 끝에 도달했거나 파일을 읽을 수 없습니다.")
            break

        result, edges = Lane_Detection(frame)
        # 결과 표시
        cv2.imshow("Lane Detection", result)
        cv2.imshow("Edges", edges)

        # 결과 비디오 저장 (선택사항)
        out.write(result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # 비디오 저장 시 필요
    cv2.destroyAllWindows()