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

def main():
    model, device = load_model(weights='/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/deep-test/weights/yolopv2_mac.pt')
    video_path = '/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/deep-test/data/main_test_video.mp4'  # 실제 비디오 경로로 수정하세요
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 크기를 720p로 조정
        frame = cv2.resize(frame, (1280, 720))
        print("Frame shape:", frame.shape)
        
        pred, da_seg_mask, ll_seg_mask = detect(model, frame, device)
        print("Pred shape:", pred[0].shape if isinstance(pred, list) else pred.shape)
        print("DA mask final shape:", da_seg_mask.shape)
        print("LL mask final shape:", ll_seg_mask.shape)
        
        # 마스크의 데이터 타입 확인 및 변환
        da_seg_mask = da_seg_mask.astype(np.uint8)
        ll_seg_mask = ll_seg_mask.astype(np.uint8)
        
        result_img = show_seg_result(frame, (da_seg_mask, ll_seg_mask), is_demo=True)
        print("Result image shape:", result_img.shape)
        
        cv2.imshow('Result', result_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()