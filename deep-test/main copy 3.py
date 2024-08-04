import cv2
import torch
import numpy as np
import os
from utils.utils import select_device, time_synchronized, scale_coords, driving_area_mask, lane_line_mask
from utils.utils import show_seg_result
from models import get_net
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

model = torch.jit.load('/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/deep-test/weights/yolopv2.pt', map_location='cpu')

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
    
    return pred, da_seg_mask, ll_seg_mask

def warp_perspective(img, src, dst, size):
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)

def extract_roi_lanes(lane_binary, src):
    mask = np.zeros_like(lane_binary)
    cv2.fillPoly(mask, [src.astype(np.int32)], 255)
    roi_lanes = cv2.bitwise_and(lane_binary, mask)
    return roi_lanes

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
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
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

def draw_lanes(img, left_fitx, right_fitx, ploty):
    out_img = np.dstack((img, img, img))
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    cv2.fillPoly(out_img, np.int_([pts]), (0, 255, 0))
    
    cv2.polylines(out_img, np.int32([pts_left]), isClosed=False, color=(255, 0, 0), thickness=15)
    cv2.polylines(out_img, np.int32([pts_right]), isClosed=False, color=(0, 0, 255), thickness=15)
    
    return out_img

def main():
    model, device = load_model(weights='/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/deep-test/weights/yolopv2_mac.pt')
    video_path = '/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/deep-test/data/main_test_video.mp4'
    cap = cv2.VideoCapture(video_path)
    
    src = np.float32([[200, 720], [1080, 720], [820, 300], [380, 300]])
    dst = np.float32([[300, 720], [980, 720], [980, 0], [300, 0]])
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (1280, 720))
        
        pred, da_seg_mask, ll_seg_mask = detect(model, frame, device)
        
        da_seg_mask = da_seg_mask.astype(np.uint8)
        ll_seg_mask = ll_seg_mask.astype(np.uint8)
        
        lane_binary = np.zeros_like(frame[:,:,0])
        lane_binary[ll_seg_mask > 0] = 255
        roi_lanes = extract_roi_lanes(lane_binary, src)
        
        bev_binary = warp_perspective(roi_lanes, src, dst, (1280, 720))
        
        left_fitx, right_fitx, ploty = fit_polynomial(bev_binary)
        lanes_img = draw_lanes(bev_binary, left_fitx, right_fitx, ploty)
        
        Minv = cv2.getPerspectiveTransform(dst, src)
        lanes_unwarped = cv2.warpPerspective(lanes_img, Minv, (1280, 720))
        
        result_img = show_seg_result(frame, (da_seg_mask, ll_seg_mask), is_demo=True)
        
        result_with_lanes = cv2.addWeighted(result_img, 1, lanes_unwarped, 0.5, 0)
        
        cv2.polylines(result_with_lanes, [src.astype(int)], True, (0, 255, 0), 2)
        
        cv2.imshow('Result with Lanes', result_with_lanes)
        cv2.imshow('ROI Lanes', roi_lanes)
        cv2.imshow("Bird's Eye View", bev_binary)
        cv2.imshow("Fitted Lanes", lanes_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()