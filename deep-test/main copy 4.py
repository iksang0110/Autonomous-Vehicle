import cv2
import torch
import numpy as np
import os
from utils.utils import select_device, time_synchronized, scale_coords, driving_area_mask, lane_line_mask
from utils.utils import show_seg_result
from models import get_net
from sklearn.linear_model import RANSACRegressor

# 모델 로드
model = torch.jit.load('/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/deep-test/weights/yolopv2.pt', map_location='cpu')

# CPU 모델로 저장
torch.jit.save(model, '/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/deep-test/weights/yolopv2_mac.pt')

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

def bird_eye_view(img, src_points=None, dst_points=None):
    h, w = img.shape[:2]
    if src_points is None:
        src_points = np.float32([
            [w * 0.43, h * 0.65],  # 좌상
            [w * 0.57, h * 0.65],  # 우상
            [w, h],                # 우하
            [0, h]                 # 좌하
        ])
    if dst_points is None:
        dst_points = np.float32([
            [w * 0.3, h * 0.1],   # 좌상 (더 좁게, 더 아래로)
            [w * 0.7, h * 0.1],   # 우상 (더 좁게, 더 아래로)
            [w * 0.9, h],         # 우하 (약간 안쪽으로)
            [w * 0.1, h]          # 좌하 (약간 안쪽으로)
        ])
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped, M

def inv_bird_eye_view(img, M):
    h, w = img.shape[:2]
    Minv = np.linalg.inv(M)
    unwarped = cv2.warpPerspective(img, Minv, (w, h), flags=cv2.INTER_LINEAR)
    return unwarped

# def additional_perspective_transform(img):
#     h, w = img.shape[:2]
#     src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
#     dst_points = np.float32([[0, h*0.2], [w, h*0.2], [w*0.8, h], [w*0.2, h]])
#     M = cv2.getPerspectiveTransform(src_points, dst_points)
#     return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

# # bird_eye_view 함수 호출 후
# warped, M = bird_eye_view(img)
# warped = additional_perspective_transform(warped)

def apply_roi(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) == 2:
        cv2.fillPoly(mask, [vertices], 255)
    else:
        cv2.fillPoly(mask, [vertices], (255, 255, 255))
    return cv2.bitwise_and(img, mask)

def apply_filters(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Sobel filter
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    
    # Threshold
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    # Morphology
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(sxbinary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    return morph

def slide_window_search(binary_warped, right_current):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    nwindows = 18
    window_height = np.int_(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    margin = 100
    minpix = 50
    right_lane = []
    color = [0, 255, 0]
    thickness = 2
    dot_save = [0] * (nwindows-3)

    for w in range(nwindows-3):
        win_y_low = binary_warped.shape[0] - (w + 1) * window_height
        win_y_high = binary_warped.shape[0] - w * window_height
        win_xright_low = right_current - margin
        win_xright_high = right_current + margin

        good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (
                nonzero_x < win_xright_high)).nonzero()[0]
        right_lane.append(good_right)

        if len(good_right) > minpix:
            right_current = np.int_(np.mean(nonzero_x[good_right]))
        else:
            w_num = w
            if w_num > 1:
                right_current = int(dot_save[w_num - 1] + (dot_save[w_num - 1] - dot_save[w_num - 2]))

        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
        dot_save[w] = right_current

    right_lane = np.concatenate(right_lane)
    rightx = nonzero_x[right_lane]
    righty = nonzero_y[right_lane]
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    rtx = np.trunc(right_fitx)
    out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]

    ym_per_pix = 98.0 / 780
    xm_per_pix = 90.0 / 600
    y_eval = np.max(ploty)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    right_curverad = 1 / right_curverad
    str__ = "curve : %f" % right_curverad
    out_img = cv2.putText(out_img, str__, (150, 40), cv2.FONT_HERSHEY_PLAIN, 2, [255, 0, 0], 1, cv2.LINE_AA)

    line_1 = right_fit[0] * ploty * 2 + right_fit[1]
    line_1 = -np.mean(line_1)

    return out_img, right_curverad, line_1, dot_save

def calculate_heading_error(left_fit, right_fit, img_shape):
    if left_fit is None and right_fit is None:
        return 0, 0, 0
    
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    
    if left_fit is not None:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    else:
        left_fitx = np.zeros_like(ploty)
    
    if right_fit is not None:
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    else:
        right_fitx = np.zeros_like(ploty)
    
    xm_per_pix = 3.7/700  # meters per pixel in x dimension
    ym_per_pix = 30/720  # meters per pixel in y dimension
    
    # 차선 중심 계산
    lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
    center_diff = (img_shape[1] / 2 - lane_center) * xm_per_pix
    
    # 곡률 반경 계산
    y_eval = np.max(ploty)
    left_curverad = 0
    right_curverad = 0
    
    if left_fit is not None:
        left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    if right_fit is not None:
        right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    # Heading error 계산
    heading_error = np.arctan(center_diff / (y_eval * ym_per_pix))
    
    return heading_error, left_curverad, right_curverad

def detect(model, img, device):
    original_img = img.copy()
    img = preprocess(img).to(device)
    
    with torch.no_grad():
        [pred, anchor_grid], seg, ll = model(img)
    
    da_seg_mask = driving_area_mask(seg)
    ll_seg_mask = lane_line_mask(ll)
    
    if isinstance(ll_seg_mask, torch.Tensor):
        ll_seg_mask = ll_seg_mask.cpu().numpy()
    if isinstance(da_seg_mask, torch.Tensor):
        da_seg_mask = da_seg_mask.cpu().numpy()
    
    # 차원을 맞추기 위해 채널 차원 추가
    if len(da_seg_mask.shape) == 2:
        da_seg_mask = da_seg_mask[..., np.newaxis]
    if len(ll_seg_mask.shape) == 2:
        ll_seg_mask = ll_seg_mask[..., np.newaxis]
    
    da_seg_mask = cv2.resize(da_seg_mask, (original_img.shape[1], original_img.shape[0]))
    ll_seg_mask = cv2.resize(ll_seg_mask, (original_img.shape[1], original_img.shape[0]))
    
    height, width = original_img.shape[:2]
    vertices = np.array([
        [(width * 0.05, height * 0.95),
         (width * 0.40, height * 0.65),
         (width * 0.60, height * 0.65),
         (width * 0.95, height * 0.95)]
    ], dtype=np.int32)
    
    roi_ll_seg_mask = apply_roi(ll_seg_mask, vertices)
    warped_ll_mask, M = bird_eye_view(roi_ll_seg_mask)
    filtered_mask = apply_filters(warped_ll_mask)
    
    histogram = np.sum(filtered_mask[filtered_mask.shape[0]//2:,:], axis=0)
    rightbase = np.argmax(histogram[:])
    
    out_img, right_curverad, line_1, dot_save = slide_window_search(filtered_mask, rightbase)
    
    return pred, da_seg_mask, roi_ll_seg_mask, warped_ll_mask, out_img, right_curverad, line_1, dot_save, M, vertices

def show_seg_result(img, result, is_demo=False):
    overlay = img.copy()
    if is_demo:
        color_area = np.zeros_like(img)
        # result[0]이 3차원이 아닌 경우 차원 추가
        if len(result[0].shape) == 2:
            result = (result[0][..., np.newaxis], result[1][..., np.newaxis])
        color_area[result[0][:,:,0] == 1] = [0, 255, 0]  # 주행 가능 영역
        color_area[result[1][:,:,0] == 1] = [255, 0, 0]  # 차선
        cv2.addWeighted(overlay, 0.7, color_area, 0.3, 0, overlay)
    return overlay
def draw_lane_lines(img, left_fit, right_fit):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    
    if left_fit is not None and right_fit is not None:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # 차선 영역을 위한 점들
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # 원본 이미지 복사
        overlay = img.copy()
        
        # 차선 영역을 투명한 녹색으로 채우기
        cv2.fillPoly(overlay, np.int_([pts]), (0, 255, 0))
        
        # 투명도 설정 (0.4 = 40% 불투명)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # 차선 경계선 그리기
        cv2.polylines(img, np.int_([pts_left]), isClosed=False, color=(255, 0, 0), thickness=2)
        cv2.polylines(img, np.int_([pts_right]), isClosed=False, color=(0, 0, 255), thickness=2)
    
    return img

def main():
    model, device = load_model(weights='/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/deep-test/weights/yolopv2_mac.pt')
    video_path = '/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/deep-test/data/main_test_video.mp4'
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (1280, 720))
        
        try:
            pred, da_seg_mask, roi_ll_seg_mask, warped_ll_mask, out_img, right_curverad, line_1, dot_save, M, vertices = detect(model, frame, device)
            
            # 원본 이미지에 차선 영역 표시
            result_img = show_seg_result(frame, (da_seg_mask, roi_ll_seg_mask), is_demo=True)
            
            # 출력 이미지 준비
            out_img = cv2.resize(out_img, (640, 480))  # 슬라이딩 윈도우 결과
            warped_ll_mask = cv2.resize(warped_ll_mask, (640, 480))  # 버드아이뷰 변환 결과
            
            # 결과 이미지 합치기
            combined_result = np.hstack((out_img, warped_ll_mask))
            
            # 곡률 및 기울기 정보 추가
            cv2.putText(combined_result, f"Curvature: {right_curverad:.4f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined_result, f"Line slope: {line_1:.4f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            # 결과 표시
            cv2.imshow('Lane Detection', combined_result)
            cv2.imshow('Segmentation Result', result_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"Error occurred: {e}")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()