import serial
import cv2
import numpy as np
from enum import Enum
import time
import math


class MissionState(Enum):
    IDLE = 0
    LANE_FOLLOWING = 1
    CROSSWALK = 2
    OBSTACLE_AVOIDANCE = 3
    PARKING = 4


class AutonomousCar:
    def __init__(self):
        # 아두이노와 시리얼 통신 설정 (포트와 baudrate는 환경에 맞게 조정 필요)
        self.arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

        # 카메라 설정 (카메라 인덱스는 시스템에 따라 다를 수 있음)
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 차량 상태 초기화
        self.state = MissionState.IDLE
        self.speed = 0
        self.steering_angle = 90  # 90도가 중앙, 0-180도 범위

        # 차선 감지를 위한 파라미터 (HSV 색상 범위)
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])

        # 신호등 감지를 위한 파라미터 (HSV 색상 범위)
        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])
        self.lower_green = np.array([40, 70, 50])
        self.upper_green = np.array([80, 255, 255])

        # PID 제어기 파라미터
        self.kp = 0.1  # 비례 게인
        self.ki = 0.01  # 적분 게인
        self.kd = 0.05  # 미분 게인
        self.previous_error = 0
        self.integral = 0

        # 주차 상태 변수
        self.parking_state = 0
        self.parking_start_time = 0

    def process_camera_input(self):
        """
        카메라로부터 이미지를 받아 처리합니다.
        :return: 처리된 프레임 또는 None (카메라 읽기 실패 시)
        """
        ret, frame = self.camera.read()
        if ret:
            # 이미지 크기 조정 (처리 속도 향상을 위해)
            return cv2.resize(frame, (320, 240))
        return None

    def detect_lane(self, frame):
        """
        주어진 프레임에서 차선을 감지합니다.
        :param frame: 입력 이미지
        :return: 차선이 표시된 프레임, 차선의 중심점 x 좌표
        """
        # HSV 색 공간으로 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 노란색 차선 마스크 생성
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # 차선의 외곽선 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 가장 큰 윤곽선 선택 (가장 뚜렷한 차선으로 가정)
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                return frame, cx

        # 차선을 찾지 못한 경우
        return frame, None

    def calculate_steering_angle(self, lane_center):
        """
        차선 중심과 화면 중심의 차이를 이용해 조향각을 계산합니다.
        :param lane_center: 차선의 중심 x 좌표
        :return: 계산된 조향각 (0-180도)
        """
        if lane_center is None:
            return 90  # 차선을 찾지 못한 경우 직진

        # 화면의 중앙과 차선 중앙의 차이 계산
        error = lane_center - 160  # 320/2 = 160 (화면 중앙)

        # PID 제어
        self.integral += error
        derivative = error - self.previous_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error

        # 출력값을 조향각으로 변환 (90도를 중심으로)
        steering_angle = 90 - output

        # 조향각 범위 제한 (0-180도)
        return max(0, min(180, steering_angle))

    def detect_traffic_light(self, frame):
        """
        프레임에서 빨간색 또는 초록색 신호등을 감지합니다.
        :param frame: 입력 이미지
        :return: 감지된 신호등 색상 ("RED", "GREEN", 또는 "NONE")
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 빨간색 마스크 (두 개의 범위를 사용)
        mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # 초록색 마스크
        mask_green = cv2.inRange(hsv, self.lower_green, self.upper_green)

        # 빨간색과 초록색 픽셀 수 계산
        red_pixels = cv2.countNonZero(mask_red)
        green_pixels = cv2.countNonZero(mask_green)

        # 임계값 설정 (이 값은 환경에 따라 조정 필요)
        threshold = 500

        if red_pixels > threshold:
            return "RED"
        elif green_pixels > threshold:
            return "GREEN"
        else:
            return "NONE"

    def detect_obstacles(self):
        """
        아두이노로부터 초음파 센서 데이터를 읽어 장애물을 감지합니다.
        :return: 장애물까지의 거리 (cm) 또는 None (데이터 읽기 실패 시)
        """
        self.arduino.write(b"GET_DISTANCE\n")
        try:
            distance_str = self.arduino.readline().decode().strip()
            distance = float(distance_str)
            return distance
        except (ValueError, serial.SerialException):
            print("Failed to read distance from Arduino")
            return None

    def control_car(self):
        """
        현재 속도와 조향각을 아두이노로 전송하여 차량을 제어합니다.
        """
        command = f"CONTROL/{self.speed}/{self.steering_angle}\n"
        try:
            self.arduino.write(command.encode())
        except serial.SerialException:
            print("Failed to send control command to Arduino")

    def lane_following(self, frame):
        """
        차선 추적 알고리즘을 실행합니다.
        :param frame: 카메라에서 받은 현재 프레임
        """
        processed_frame, lane_center = self.detect_lane(frame)
        self.steering_angle = self.calculate_steering_angle(lane_center)

        # 기본 속도 설정 (환경에 따라 조정 필요)
        self.speed = 150

        # 커브가 심한 경우 속도 감소
        if abs(self.steering_angle - 90) > 30:
            self.speed = 100

    def handle_crosswalk(self, frame):
        """
        횡단보도와 신호등을 처리합니다.
        :param frame: 카메라에서 받은 현재 프레임
        """
        traffic_light = self.detect_traffic_light(frame)
        if traffic_light == "RED":
            self.speed = 0  # 빨간 신호에서 정지
        elif traffic_light == "GREEN":
            self.speed = 150  # 초록 신호에서 출발
            self.state = MissionState.LANE_FOLLOWING  # 차선 추적으로 상태 변경
        else:
            self.speed = 50  # 신호등이 없는 경우 서행

    def avoid_obstacles(self):
        """
        장애물을 감지하고 회피합니다.
        """
        distance = self.detect_obstacles()
        if distance is not None:
            if distance < 20:  # 20cm 이내에 장애물
                self.speed = 0
                self.steering_angle = 135  # 오른쪽으로 45도 회전
            elif distance < 50:  # 50cm 이내에 장애물
                self.speed = 100
                self.steering_angle = 110  # 오른쪽으로 약간 회전
            else:
                self.speed = 150
                self.steering_angle = 90  # 직진

    def park_car(self):
        """
        주차 알고리즘을 실행합니다.
        """
        current_time = time.time()

        if self.parking_state == 0:
            # 주차 공간 탐색
            self.speed = 50
            self.steering_angle = 90
            distance = self.detect_obstacles()
            if distance is not None and distance > 100:  # 주차 공간 발견
                self.parking_state = 1
                self.parking_start_time = current_time

        elif self.parking_state == 1:
            # 주차 공간 통과
            if current_time - self.parking_start_time < 2:
                self.speed = 50
                self.steering_angle = 90
            else:
                self.parking_state = 2
                self.parking_start_time = current_time

        elif self.parking_state == 2:
            # 후진 및 조향
            if current_time - self.parking_start_time < 2:
                self.speed = -50
                self.steering_angle = 135
            else:
                self.parking_state = 3
                self.parking_start_time = current_time

        elif self.parking_state == 3:
            # 후진 및 반대 방향 조향
            if current_time - self.parking_start_time < 2:
                self.speed = -50
                self.steering_angle = 45
            else:
                self.parking_state = 4

        elif self.parking_state == 4:
            # 주차 완료
            self.speed = 0
            self.steering_angle = 90
            print("Parking completed")
            self.state = MissionState.IDLE

    def run(self):
        """
        메인 루프: 모든 기능을 실행하고 차량을 제어합니다.
        """
        while True:
            frame = self.process_camera_input()
            if frame is None:
                continue

            if self.state == MissionState.LANE_FOLLOWING:
                self.lane_following(frame)
            elif self.state == MissionState.CROSSWALK:
                self.handle_crosswalk(frame)
            elif self.state == MissionState.OBSTACLE_AVOIDANCE:
                self.avoid_obstacles()
            elif self.state == MissionState.PARKING:
                self.park_car()

            self.control_car()

            # 상태 전환 로직
            distance = self.detect_obstacles()
            if distance is not None and distance < 100:
                self.state = MissionState.OBSTACLE_AVOIDANCE
            elif self.detect_traffic_light(frame) != "NONE":
                self.state = MissionState.CROSSWALK

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    car = AutonomousCar()
    car.run()