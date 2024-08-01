import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import scipy.special
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
from utils.config import Config
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Lane Detection')
    parser.add_argument('config', help='/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/Ultra-Fast-Lane-Detection-master/configs/tusimple.py')
    parser.add_argument('--video_path', help='/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/drive3.mp4', required=True)
    parser.add_argument('--test_model', help='/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/Ultra-Fast-Lane-Detection-master/model/tusimple_18.pth', required=True)
    return parser.parse_args()

def resize_image(img, size):
    return cv2.resize(img, (size[1], size[0]))

def transform_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_image(img, (288, 800))
    img = transforms.ToTensor()(img)
    img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
    return img

def get_lanes(output, cfg):
    col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    out_j = output[0].data.cpu().numpy()
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(cfg.griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == cfg.griding_num] = 0
    out_j = loc

    lanes = []
    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            lane = []
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    lane.append((int(out_j[k, i] * col_sample_w * 1280 / 800) - 1, int(720 * (cfg.row_anchor[cfg.cls_num_per_lane-1-k]/288)) - 1 ))
            lanes.append(lane)
    return lanes

def draw_lanes(img, lanes):
    for lane in lanes:
        for i in range(len(lane) - 1):
            cv2.line(img, lane[i], lane[i+1], (0, 255, 0), 2)
    return img

if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.video_path = args.video_path
    cfg.test_model = args.test_model

    # 필요한 설정값 확인 및 설정
    if not hasattr(cfg, 'data_root'):
        cfg.data_root = ''
    if not hasattr(cfg, 'log_path'):
        cfg.log_path = ''
    if not hasattr(cfg, 'finetune'):
        cfg.finetune = None
    if not hasattr(cfg, 'resume'):
        cfg.resume = None
    if not hasattr(cfg, 'test_work_dir'):
        cfg.test_work_dir = ''

    # row_anchor 설정 추가
    if cfg.dataset == 'CULane':
        cfg.row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
    elif cfg.dataset == 'Tusimple':
        cfg.row_anchor = [64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284]
    else:
        raise NotImplementedError

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cfg.cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cfg.cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(pretrained = False, backbone=cfg.backbone, cls_dim = (cfg.griding_num+1,cfg.cls_num_per_lane,4),
                    use_aux=False).eval()

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)

    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        img = transform_image(frame)
        img = img.unsqueeze(0)

        with torch.no_grad():
            out_j = net(img)

        lanes = get_lanes(out_j, cfg)
        frame_with_lanes = draw_lanes(frame, lanes)

        cv2.imshow('Lane Detection', frame_with_lanes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    cv2.destroyAllWindows()

    print(f"처리가 완료되었습니다. 총 {frame_count} 프레임을 처리했습니다.")