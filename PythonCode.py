import cv2
from collections import deque
from ultralytics import YOLO
import socket
import json
import math

UDP_IP = "192.168.0.210"
UDP_PORT = 5005

parking_spots = [
    {"x": 2,   "y": 9,  "width": 5, "length": 10, "angle": math.radians(-30)},
    {"x": -4,  "y": 11, "width": 5, "length": 10, "angle": math.radians(-30)},
    {"x": -10, "y": 13, "width": 5, "length": 10, "angle": math.radians(-30)},
    {"x": 6,   "y": 9,  "width": 5, "length": 10, "angle": math.radians(25)},
    {"x": 10,  "y": 11, "width": 5, "length": 10, "angle": math.radians(25)},
]

CAMERA_HEIGHT   = 4
REAL_CAR_LENGTH = 4.5
REAL_CAR_WIDTH  = 1.8
FOCAL_FACTOR    = 800
HISTORY_LEN     = 15
MAX_JUMP_M      = 1.5

sock  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
model = YOLO("yolov10x.pt")
vehicle_classes = {2, 3, 5, 7}


def get_rect_corners(center_x, center_y, width, length, angle=0.0):
    w2, h2 = length / 2, width / 2
    corners = [(-w2, -h2), (-w2, h2), (w2, h2), (w2, -h2)]
    rotated = []
    for x, y in corners:
        xr = x * math.cos(angle) - y * math.sin(angle)
        yr = x * math.sin(angle) + y * math.cos(angle)
        rotated.append((center_x + xr, center_y + yr))
    return rotated


def process_video():
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("❌ Камера не найдена")
        return

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print(f"✅ Старт на {UDP_IP}:{UDP_PORT}")

    # Парковки считаем один раз
    flat_spots = []
    for spot in parking_spots:
        corners = get_rect_corners(
            spot["x"], spot["y"],
            spot["width"], spot["length"],
            spot["angle"]
        )
        flat = []
        for x, y in corners:
            flat.extend([round(x, 3), round(y, 3)])
        flat_spots.append(flat)

    # ← Вынесено из цикла!
    prev_widths      = {}
    position_history = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, verbose=False)[0]
        detected_rects = []

        for box in results.boxes:
            cls = int(box.cls[0])
            if cls not in vehicle_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w_px_raw = x2 - x1

            track_id = int(box.id[0]) if box.id is not None else id(box)

            # Сглаживание ширины бокса
            prev_w = prev_widths.get(track_id, w_px_raw)
            w_px = 0.7 * prev_w + 0.3 * w_px_raw
            prev_widths[track_id] = w_px

            if w_px <= 10:
                continue

            meters_per_px = REAL_CAR_LENGTH / w_px
            cx_px         = (x1 + x2) / 2
            offset_x_m    = (cx_px - frame_width / 2) * meters_per_px

            perspective_correction = 1 + abs(offset_x_m) * 0.05
            camera_dist = (REAL_CAR_LENGTH * FOCAL_FACTOR) / (w_px * perspective_correction)
            car_y_dist  = math.sqrt(max(camera_dist ** 2 - CAMERA_HEIGHT ** 2, 0.01))
            car_y_dist  = max(car_y_dist, 0.5)

            # Защита от прыжков
            if track_id in position_history and len(position_history[track_id]) > 0:
                last = position_history[track_id][-1]
                jump = math.sqrt((offset_x_m - last[0])**2 + (car_y_dist - last[1])**2)
                if jump > MAX_JUMP_M:
                    offset_x_m, car_y_dist = last

            # История позиций
            if track_id not in position_history:
                position_history[track_id] = deque(maxlen=HISTORY_LEN)
            position_history[track_id].append((offset_x_m, car_y_dist))

            # Усреднение
            avg_x = sum(p[0] for p in position_history[track_id]) / len(position_history[track_id])
            avg_y = sum(p[1] for p in position_history[track_id]) / len(position_history[track_id])

            corners = get_rect_corners(avg_x, avg_y, REAL_CAR_WIDTH, REAL_CAR_LENGTH)
            flat_corners = []
            for x, y in corners:
                flat_corners.extend([round(x, 3), round(y, 3)])

            hypot_dist = math.sqrt(avg_x ** 2 + avg_y ** 2)
            flat_corners.append(round(hypot_dist, 2))
            detected_rects.append(flat_corners)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{hypot_dist:.1f}m", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        data = {"rectangles": detected_rects, "parking_spots": flat_spots}
        try:
            sock.sendto(json.dumps(data).encode("utf-8"), (UDP_IP, UDP_PORT))
        except Exception as e:
            print("UDP ошибка:", e)

        cv2.imshow("Parking AI", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    sock.close()


process_video()