import cv2
import numpy as np
import darknet
import time

# YOLO 설정
config_path = "yolov3-tiny.cfg"
weights_path = "yolov3-tiny.weights"
data_file = "coco.data"

# 네트워크, 클래스, 컬러 로드
network, class_names, class_colors = darknet.load_network(
    config_path,
    data_file,
    weights_path,
    batch_size=1
)

def convert2relative(bbox):
    x, y, w, h = bbox
    _height = darknet.network_height(network)
    _width = darknet.network_width(network)
    return x/_width, y/_height, w/_width, h/_height

def convert2original(image, bbox):
    x, y, w, h = bbox
    image_h, image_w, _ = image.shape
    orig_x = int(x * image_w)
    orig_y = int(y * image_h)
    orig_w = int(w * image_w)
    orig_h = int(h * image_h)
    return (orig_x, orig_y, orig_w, orig_h)

def draw_boxes(detections, img):
    for detection in detections:
        x, y, w, h = convert2original(img, detection[2])
        pt1 = (x - w//2, y - h//2)
        pt2 = (x + w//2, y + h//2)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img, detection[0].decode() + " [" + str(round(detection[1] * 100, 2)) + "]", (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)
    return img

# 비디오 로드
video_path = "/Users/02.011x/Documents/GitHub/Autonomous-Vehicle/testvideo.mp4"
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    prev_time = time.time()

    # YOLO 모델을 위한 이미지 전처리
    darknet_image = darknet.make_image(frame_width, frame_height, 3)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.25)
    darknet.free_image(darknet_image)
    
    # 객체 검출 결과를 이미지에 표시
    frame = draw_boxes(detections, frame)

    # 차선 검출 로직
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

    # 결과 비디오 작성
    out.write(frame)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
