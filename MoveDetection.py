import cv2
import numpy as np
from ultralytics import YOLO
import sqlite3
import os
from datetime import datetime

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

# 이미지 저장을 위한 디렉토리 생성
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)


def detect_people(frame):
    results = model(frame)
    return results[0].boxes.xyxy.cpu().numpy()


def estimate_distance(bbox, image_height):
    return 1 / (bbox[3] - bbox[1]) * image_height


def image_to_vehicle_coords(bbox, image_shape):
    center_x = (bbox[0] + bbox[2]) / 2
    bottom_y = bbox[3]

    x = (center_x - image_shape[1] / 2) / camera_matrix[0, 0]
    y = (bottom_y - image_shape[0] / 2) / camera_matrix[1, 1]

    return x, y


# 비디오 캡처 (실제로는 차량 카메라 스트림을 사용해야 합니다)
cap = cv2.VideoCapture(0)

frame_count = 0
save_interval = 30  # 30프레임마다 이미지 저장

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

    # 일정 간격으로 이미지 저장
    if frame_count % save_interval == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/frame_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Image saved: {filename}")

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()