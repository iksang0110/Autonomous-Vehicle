#include <Servo.h>

// 모터 제어 핀 정의
#define MOTOR_LEFT_FORWARD 4
#define MOTOR_LEFT_BACKWARD 5
#define MOTOR_RIGHT_FORWARD 6
#define MOTOR_RIGHT_BACKWARD 7
#define SERVO_PIN 9

// 초음파 센서 핀 정의
#define TRIG_PIN 11
#define ECHO_PIN 12

// 서보 모터 객체 생성
Servo steering_servo;

// 전역 변수 선언
int current_speed = 0;
int current_angle = 90;

void setup() {
  // 시리얼 통신 초기화 (115200bps로 설정, 더 빠른 통신을 위해)
  Serial.begin(115200);
  
  // 모터 제어 핀을 출력으로 설정
  pinMode(MOTOR_LEFT_FORWARD, OUTPUT);
  pinMode(MOTOR_LEFT_BACKWARD, OUTPUT);
  pinMode(MOTOR_RIGHT_FORWARD, OUTPUT);
  pinMode(MOTOR_RIGHT_BACKWARD, OUTPUT);
  
  // 초음파 센서 핀 설정
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  
  // 서보 모터 초기화 및 중앙 위치(90도)로 설정
  steering_servo.attach(SERVO_PIN);
  steering_servo.write(90);
  
  // 초기화 완료 메시지
  Serial.println("Arduino initialized and ready.");
}

void loop() {
  // 시리얼 통신으로부터 명령 읽기
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    processCommand(command);
  }
}

void processCommand(String command) {
  // 명령 형식: "CONTROL/speed/angle" 또는 "GET_DISTANCE"
  if (command.startsWith("CONTROL/")) {
    // CONTROL 명령 처리
    int separatorIndex = command.indexOf('/', 8);
    if (separatorIndex != -1) {
      int speed = command.substring(8, separatorIndex).toInt();
      int angle = command.substring(separatorIndex + 1).toInt();
      setMotorSpeed(speed);
      setSteeringAngle(angle);
    }
  } else if (command == "GET_DISTANCE") {
    // 거리 측정 명령 처리
    float distance = measureDistance();
    Serial.println(distance);
  }
}

void setMotorSpeed(int speed) {
  // 속도 범위: -255 ~ 255
  // 양수: 전진, 음수: 후진, 0: 정지
  speed = constrain(speed, -255, 255);
  
  if (speed > 0) {
    // 전진
    analogWrite(MOTOR_LEFT_FORWARD, speed);
    analogWrite(MOTOR_RIGHT_FORWARD, speed);
    analogWrite(MOTOR_LEFT_BACKWARD, 0);
    analogWrite(MOTOR_RIGHT_BACKWARD, 0);
  } else if (speed < 0) {
    // 후진
    analogWrite(MOTOR_LEFT_FORWARD, 0);
    analogWrite(MOTOR_RIGHT_FORWARD, 0);
    analogWrite(MOTOR_LEFT_BACKWARD, -speed);
    analogWrite(MOTOR_RIGHT_BACKWARD, -speed);
  } else {
    // 정지
    analogWrite(MOTOR_LEFT_FORWARD, 0);
    analogWrite(MOTOR_RIGHT_FORWARD, 0);
    analogWrite(MOTOR_LEFT_BACKWARD, 0);
    analogWrite(MOTOR_RIGHT_BACKWARD, 0);
  }
  
  current_speed = speed;
}

void setSteeringAngle(int angle) {
  // 조향각 범위: 0 ~ 180도
  // 90도: 직진, <90: 좌회전, >90: 우회전
  angle = constrain(angle, 0, 180);
  steering_servo.write(angle);
  current_angle = angle;
}

float measureDistance() {
  // 초음파 센서를 사용하여 거리 측정
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  
  // 에코 핀으로부터 펄스 지속 시간 측정
  long duration = pulseIn(ECHO_PIN, HIGH);
  
  // 거리 계산 (음속: 340m/s, 왕복 거리이므로 2로 나눔)
  float distance = (duration * 0.0343) / 2;
  
  return distance;
}

// 디버깅을 위한 현재 상태 출력 함수
void printStatus() {
  Serial.print("Speed: ");
  Serial.print(current_speed);
  Serial.print(", Angle: ");
  Serial.println(current_angle);
}
