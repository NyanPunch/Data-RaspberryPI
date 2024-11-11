import cv2
import numpy as np
import torch
from ultralytics import YOLO  # YOLO 모델 import (실제로는 적절한 YOLO 라이브러리를 사용해야 합니다)
import sqlite3

# YOLO 모델 로드
model = YOLO('yolov5s.pt')

# 카메라 캘리브레이션 매트릭스 (예시 값, 실제 카메라에 맞게 조정 필요)
camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])

# 데이터베이스 연결
conn = sqlite3.connect('detections.db')
cursor = conn.cursor()

# 테이블 생성
cursor.execute('''CREATE TABLE IF NOT EXISTS detections
                  (timestamp TEXT, x REAL, y REAL, distance REAL)''')


def detect_people(frame):
    results = model(frame)
    return results[0].boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표 반환


def estimate_distance(bbox, image_height):
    # 간단한 거리 추정 (실제로는 더 복잡한 계산이 필요할 수 있습니다)
    return 1 / (bbox[3] - bbox[1]) * image_height


def image_to_vehicle_coords(bbox, image_shape):
    # 이미지 좌표를 차량 기준 좌표로 변환 (간단한 예시)
    center_x = (bbox[0] + bbox[2]) / 2
    bottom_y = bbox[3]

    x = (center_x - image_shape[1] / 2) / camera_matrix[0, 0]
    y = (bottom_y - image_shape[0] / 2) / camera_matrix[1, 1]

    return x, y


# 비디오 캡처 (실제로는 차량 카메라 스트림을 사용해야 합니다)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_people(frame)

    for bbox in detections:
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        distance = estimate_distance(bbox, frame.shape[0])
        x, y = image_to_vehicle_coords(bbox, frame.shape)

        # 데이터베이스에 저장
        cursor.execute("INSERT INTO detections VALUES (datetime('now'), ?, ?, ?)",
                       (x, y, distance))
        conn.commit()

        # 화면에 정보 표시
        cv2.putText(frame, f"Distance: {distance:.2f}m", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
