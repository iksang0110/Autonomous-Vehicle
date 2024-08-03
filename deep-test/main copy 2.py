import cv2
import torch
import numpy as np
import os
from scipy import stats
from utils.utils import select_device, time_synchronized, scale_coords, driving_area_mask, lane_line_mask
from utils.utils import show_seg_result
from models import get_net

# 모델 로드
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def load_model(weights='/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/deep-test/weights/yolopv2.pt', device='mps'):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = torch.jit.load(weights, map_location=device)
    model.eval()
    return model, device

def preprocess(img):
    img = cv2.resize(img, (640, 384))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def detect(model, img, device):
    img = preprocess(img).to(device)
    
    with torch.no_grad():
        [pred, anchor_grid], seg, ll = model(img)
    
    da_seg_mask = driving_area_mask(seg)
    ll_seg_mask = lane_line_mask(ll)
    
    return pred, da_seg_mask, ll_seg_mask

def bird_eye_view(img, src, dst):
    h, w = img.shape[:2]
    src = src.astype(np.float32)
    dst = dst.astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    
    # float 이미지를 uint8로 변환
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    
    # 결과를 다시 0-1 범위의 float으로 변환
    if img.dtype == np.float32 or img.dtype == np.float64:
        warped = warped.astype(float) / 255.0
    
    return warped

def histogram(img):
    return np.sum(img[img.shape[0]//2:,:], axis=0)

def find_lane_pixels(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    margin = 100
    minpix = 50

    window_height = int(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def fit_polynomial(binary_warped):
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty

def ransac_polyfit(x, y, order=2):
    model = np.poly1d(np.polyfit(x, y, order))
    
    n_samples = len(x)
    n_iterations = 100
    threshold = 5
    min_samples = order + 1
    
    best_score = 0
    best_model = None
    
    for _ in range(n_iterations):
        sample_indices = np.random.choice(n_samples, min_samples, replace=False)
        sample_x = x[sample_indices]
        sample_y = y[sample_indices]
        
        sample_model = np.poly1d(np.polyfit(sample_x, sample_y, order))
        
        distances = np.abs(sample_model(x) - y)
        inliers = distances < threshold
        n_inliers = np.sum(inliers)
        
        if n_inliers > best_score:
            best_score = n_inliers
            best_model = sample_model
    
    return best_model

def post_process_lane(lane_mask):
    # float 타입을 uint8로 변환
    lane_mask = (lane_mask * 255).astype(np.uint8)

    kernel = np.ones((5,1), np.uint8)
    lane_mask = cv2.erode(lane_mask, kernel, iterations=1)
    lane_mask = cv2.dilate(lane_mask, kernel, iterations=1)
    
    kernel = np.ones((3,3), np.uint8)
    lane_mask = cv2.erode(lane_mask, kernel, iterations=1)
    
    # 결과를 다시 0-1 범위의 float으로 변환
    return lane_mask.astype(float) / 255.0

def apply_sobel_and_morphology(img):
    # 입력 이미지가 이미 그레이스케일인 경우 처리
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        gray = img.squeeze() if len(img.shape) == 3 else img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # float 이미지를 uint8로 변환
    if gray.dtype == np.float32 or gray.dtype == np.float64:
        gray = (gray * 255).astype(np.uint8)
    
    # Sobel 필터 적용
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    
    # 이진화
    thresh_min = 20
    thresh_max = 100
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    # Morphology 연산
    kernel = np.ones((3,3), np.uint8)
    binary_output = cv2.morphologyEx(binary_output, cv2.MORPH_CLOSE, kernel)
    
    return binary_output

def cluster_points(points, max_distance=10):
    clusters = []
    for point in points:
        if not clusters:
            clusters.append([point])
        else:
            min_distance = float('inf')
            closest_cluster = None
            for cluster in clusters:
                distance = np.min(np.linalg.norm(np.array(cluster) - point, axis=1))
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = cluster
            
            if min_distance <= max_distance:
                closest_cluster.append(point)
            else:
                clusters.append([point])
    
    return clusters

def calculate_heading_error(left_fit, right_fit, y_eval):
    left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    
    lane_center = (left_x + right_x) / 2
    image_center = 640 / 2
    
    heading_error = lane_center - image_center
    
    return heading_error

def draw_roi(img, points):
    """
    이미지에 ROI 영역을 그립니다.
    
    :param img: 원본 이미지
    :param points: ROI의 꼭지점 좌표 (4개의 점)
    :return: ROI가 그려진 이미지
    """
    # 이미지 복사
    roi_img = img.copy()
    
    # 다각형 그리기
    cv2.polylines(roi_img, [points.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # 반투명한 다각형 채우기
    overlay = roi_img.copy()
    cv2.fillPoly(overlay, [points.astype(np.int32)], color=(0, 255, 0))
    
    # 원본 이미지와 오버레이 이미지를 합성
    output = cv2.addWeighted(roi_img, 0.7, overlay, 0.3, 0)
    
    return output

def main():
    model, device = load_model(weights='/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/deep-test/weights/yolopv2_mac.pt')
    video_path = '/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/deep-test/data/main_test_video.mp4'
    cap = cv2.VideoCapture(video_path)

    h, w = 720, 1280  # 프레임 크기에 맞게 조정
    src = np.array([
        [w * 0.43, h * 0.65],  # 좌상단
        [w * 0.57, h * 0.65],  # 우상단
        [w * 0.1, h * 0.95],   # 좌하단
        [w * 0.9, h * 0.95]    # 우하단
    ], dtype=np.float32)

    dst = np.array([
        [0.25*w, 0],      # 좌상단
        [0.75*w, 0],      # 우상단
        [0.25*w, h],      # 좌하단
        [0.75*w, h]       # 우하단
    ], dtype=np.float32)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (1280, 720))
        
        pred, da_seg_mask, ll_seg_mask = detect(model, frame, device)
        
         # ROI 영역 그리기
        roi_frame = draw_roi(frame, src)
        cv2.imshow('ROI', roi_frame)

        cv2.imshow('Original Frame', frame)
        
        result_img = show_seg_result(frame.copy(), (da_seg_mask, ll_seg_mask), is_demo=True)
        cv2.imshow('YOLOPv2 Result', result_img)
        
        ll_seg_mask = post_process_lane(ll_seg_mask)
        cv2.imshow('Post-processed Lane Mask', (ll_seg_mask * 255).astype(np.uint8))
        
        warped = bird_eye_view(ll_seg_mask, src, dst)
        cv2.imshow('Bird\'s Eye View', warped * 255)
        
        binary_warped = apply_sobel_and_morphology(warped)
        cv2.imshow('Sobel and Morphology', binary_warped * 255)
        
        left_fitx, right_fitx, ploty = fit_polynomial(binary_warped)
        
        left_fit = ransac_polyfit(ploty, left_fitx)
        right_fit = ransac_polyfit(ploty, right_fitx)
        
        left_points = np.column_stack((left_fitx, ploty))
        right_points = np.column_stack((right_fitx, ploty))
        left_clusters = cluster_points(left_points)
        right_clusters = cluster_points(right_points)
        
        heading_error = calculate_heading_error(left_fit, right_fit, frame.shape[0])
        
        final_result = frame.copy()
        cv2.polylines(final_result, [np.column_stack((left_fitx, ploty)).astype(np.int32)], isClosed=False, color=(255,0,0), thickness=2)
        cv2.polylines(final_result, [np.column_stack((right_fitx, ploty)).astype(np.int32)], isClosed=False, color=(0,0,255), thickness=2)
        cv2.putText(final_result, f'Heading Error: {heading_error:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Final Result', final_result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()