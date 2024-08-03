import cv2
import torch
import numpy as np
import os
from utils.utils import select_device, time_synchronized, scale_coords, driving_area_mask, lane_line_mask
from utils.utils import show_seg_result
from models import get_net
import torch

# 모델 로드
model = torch.jit.load('/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/deep-test/weights/yolopv2.pt', map_location='cpu')

# CPU 모델로 저장
torch.jit.save(model, '/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/deep-test/weights/yolopv2_mac.pt')

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# def load_model(weights='/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/deep-test/weights/yolopv2.pt', device=''):
#     device = select_device(device)
#     weights = os.path.join(os.path.dirname(__file__), weights)
#     if not os.path.exists(weights):
#         raise FileNotFoundError(f"Weights file not found: {weights}")
#     model = torch.jit.load(weights)
#     if device.type != 'cpu':
#         model = model.to(device)
#     model.eval()
#     return model, device

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

# def detect(model, img, device):
#     img = preprocess(img).to(device)
    
#     with torch.no_grad():
#         [pred, anchor_grid], seg, ll = model(img)
        
#     da_seg_mask = driving_area_mask(seg)
#     ll_seg_mask = lane_line_mask(ll)
    
#     return pred, da_seg_mask, ll_seg_mask
def detect(model, img, device):
    img = preprocess(img).to(device)
    
    with torch.no_grad():
        [pred, anchor_grid], seg, ll = model(img)
    
    print("seg shape:", seg.shape)
    print("ll shape:", ll.shape)
    print("seg min-max:", seg.min().item(), seg.max().item())
    print("ll min-max:", ll.min().item(), ll.max().item())
    
    da_seg_mask = driving_area_mask(seg)
    ll_seg_mask = lane_line_mask(ll)
    
    print("da_seg_mask shape:", da_seg_mask.shape)
    print("ll_seg_mask shape:", ll_seg_mask.shape)
    print("da_seg_mask min-max:", da_seg_mask.min(), da_seg_mask.max())
    print("ll_seg_mask min-max:", ll_seg_mask.min(), ll_seg_mask.max())
    
    # 마스크가 이미 NumPy 배열이므로 추가 변환 불필요
    
    return pred, da_seg_mask, ll_seg_mask

def warp_perspective(img, src, dst, size):
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)

def apply_sobel_and_morphology(img):

    print("Input image shape:", img.shape)
    print("Input image dtype:", img.dtype)
    print("Input image min-max:", np.min(img), np.max(img))

    # Sobel 필터 적용 (수평 방향)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.absolute(sobel_x)
    sobel_x = np.uint8(255 * sobel_x / np.max(sobel_x))

    # 이진화
    _, binary = cv2.threshold(sobel_x, 20, 255, cv2.THRESH_BINARY)

    # Morphology 연산
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=2)

    print("Output image min-max:", np.min(dilated), np.max(dilated))
    return dilated

def main():
    model, device = load_model(weights='/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/deep-test/weights/yolopv2_mac.pt')
    video_path = '/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/deep-test/data/main_test_video.mp4'
    cap = cv2.VideoCapture(video_path)
    
    src = np.float32([[200, 720], [1080, 720], [820, 300], [380, 300]])
    dst = np.float32([[300, 720], [980, 720], [1280, 0], [0, 0]])
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (1280, 720))
        
        pred, da_seg_mask, ll_seg_mask = detect(model, frame, device)
        
        da_seg_mask = da_seg_mask.astype(np.uint8)
        ll_seg_mask = ll_seg_mask.astype(np.uint8)
        
        # 차선 마스크 이진화
        lane_binary = np.zeros_like(frame[:,:,0])
        lane_binary[ll_seg_mask > 0] = 255
        
        # Bird's eye view 변환
        bev_binary = warp_perspective(lane_binary, src, dst, (1280, 720))
        
        # Sobel 및 Morphology 필터 적용
        filtered_bev = apply_sobel_and_morphology(bev_binary)
        
        # 결과 이미지 생성
        result_img = show_seg_result(frame, (da_seg_mask, ll_seg_mask), is_demo=True)
        
        # ROI 영역 표시
        roi_display = result_img.copy()
        cv2.polylines(roi_display, [src.astype(int)], True, (0, 255, 0), 2)
        
        # 화면 표시
        cv2.imshow('Result with ROI', roi_display)
        cv2.imshow('Lane Binary', lane_binary)
        cv2.imshow("Bird's Eye View", bev_binary)
        cv2.imshow("Filtered Bird's Eye View", filtered_bev)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()