#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from ultralytics import YOLO
import RPi.GPIO as GPIO
from collections import deque, Counter
import threading
from queue import Queue, Empty

# --- Configuration ---
GPIO.setwarnings(False)
GPIO_MODE = True

@dataclass
class Config:
    model_path: str = "/home/rpi/yolo/yolo11n.pt"
    source: any = 0
    resolution: tuple = (640, 480)
    
    # [OPTIMIZED] เพิ่มค่า conf_thresh เพื่อกรองผลลัพธ์ที่ไม่ชัดเจนออกไปเร็วขึ้น
    conf_thresh: float = 0.12
    
    # [OPTIMIZED] ลดขนาด imgsz ลงอย่างมาก ซึ่งเป็นวิธีที่ได้ผลที่สุดในการเพิ่ม FPS
    # คุณสามารถทดลองปรับค่านี้ได้ระหว่าง 160, 192, 224 เพื่อหาจุดสมดุลระหว่างความเร็วและความแม่นยำ
    imgsz: int = 256
    
    steering_gain: float = 0.02
    max_angle: float = 30.0
    roi_top_ratio: float = 0.5
    canny_low: int = 50
    canny_high: int = 150
    focal_length: float = 1500.0
    normal_speed: int = 30
    detection_distance_m: float = 75.0
    lane_origin_y_ratio: float = 0.75
    
    TARGET_CLASSES: list = field(default_factory=lambda: ["person", "car", "truck", "bus", "bicycle", "motorcycle"])
    
    # --- Calibration & Tuning ---
    LANE_DIST_CALIB_M: tuple = (3, 80)
    lane_mask_margin: int = 15
    OBJECT_REAL_HEIGHTS: dict = field(default_factory=lambda: {
        "person": 1.7, "car": 1.5, "truck": 3.5, "bus": 3.2, "bicycle": 1.0, "motorcycle": 1.2
    })
    DEFAULT_OBJECT_HEIGHT: float = 1.5

# --- MotorControl Class ---
class MotorControl:
    def __init__(self, pin_b, pin_c, freq=50):
        self.pin_b = pin_b
        self.pin_c = pin_c
        self.freq = freq
        self.servo_pwm1 = None
        self.servo_pwm2 = None
        self.current_angle = 0.0
        if GPIO_MODE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin_b, GPIO.OUT)
            GPIO.setup(self.pin_c, GPIO.OUT)
            self.servo_pwm1 = GPIO.PWM(self.pin_b, self.freq)
            self.servo_pwm2 = GPIO.PWM(self.pin_c, self.freq)
            self.servo_pwm1.start(0)
            self.servo_pwm2.start(0)

    def _angle_to_duty(self, angle: float) -> float:
        return 7.5 + (angle / 30.0) * 2.0

    def move_to(self, angle: float):
        self.current_angle = np.clip(angle, -30.0, 30.0)
        duty = self._angle_to_duty(self.current_angle)
        if GPIO_MODE:
            self.servo_pwm1.ChangeDutyCycle(duty)
            self.servo_pwm2.ChangeDutyCycle(duty)

    def set_stop(self):
        if GPIO_MODE:
            self.servo_pwm1.ChangeDutyCycle(0)
            self.servo_pwm2.ChangeDutyCycle(0)
        self.current_angle = 0.0

    def stop(self):
        if GPIO_MODE:
            self.servo_pwm1.stop()
            self.servo_pwm2.stop()
            GPIO.cleanup()

# --- LaneDetector Class ---
class LaneDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.left_lanes = deque(maxlen=10)
        self.right_lanes = deque(maxlen=10)

    def meters_to_y(self, meters: float, height: int) -> int:
        m_near, m_far = self.cfg.LANE_DIST_CALIB_M
        y_near, y_far = height, int(height * self.cfg.roi_top_ratio)
        y_coord = np.interp(meters, [m_near, m_far], [y_near, y_far])
        return int(np.clip(y_coord, y_far, y_near))

    def _average_lines(self, lines, height):
        if not lines: return None
        avg_line = np.mean(np.array(lines), axis=0, dtype=np.int32)
        x1, y1, x2, y2 = avg_line
        if x1 == x2: return None
        slope = (y2 - y1) / (x2 - x1)
        if slope == 0: return None
        intercept = y1 - slope * x1
        
        y1_new = int(height * self.cfg.lane_origin_y_ratio)
        y2_new = self.meters_to_y(self.cfg.detection_distance_m, height)
        
        x1_new = int((y1_new - intercept) / slope)
        x2_new = int((y2_new - intercept) / slope)
        return (x1_new, y1_new, x2_new, y2_new)

    def detect_lanes(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.cfg.canny_low, self.cfg.canny_high)
        h, w = edges.shape
        mask = np.zeros_like(edges)
        pts = np.array([[(0, h), (int(w * 0.45), int(h * self.cfg.roi_top_ratio)), (int(w * 0.55), int(h * self.cfg.roi_top_ratio)), (w, h)]], np.int32)
        cv2.fillPoly(mask, pts, 255)
        roi = cv2.bitwise_and(edges, mask)
        lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 20, minLineLength=20, maxLineGap=30)
        if lines is None:
            if self.left_lanes and self.right_lanes:
                return [np.mean(self.left_lanes, axis=0, dtype=np.int32), np.mean(self.right_lanes, axis=0, dtype=np.int32)]
            return []
        left, right = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1: continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3: continue
            (left if slope < 0 else right).append((x1, y1, x2, y2))
        if avg_left := self._average_lines(left, h): self.left_lanes.append(avg_left)
        if avg_right := self._average_lines(right, h): self.right_lanes.append(avg_right)
        lanes = []
        if self.left_lanes: lanes.append(np.mean(self.left_lanes, axis=0, dtype=np.int32))
        if self.right_lanes: lanes.append(np.mean(self.right_lanes, axis=0, dtype=np.int32))
        return lanes

    def draw_lanes(self, frame, lanes):
        if not lanes: return frame
        lane_img = np.zeros_like(frame)
        for x1, y1, x2, y2 in lanes:
            cv2.line(lane_img, (x1, y1), (x2, y2), (0, 255, 255), 4)
        return cv2.addWeighted(frame, 1.0, lane_img, 1.0, 0)

    def get_lane_area_mask(self, shape, lanes):
        if len(lanes) != 2: return np.zeros(shape[:2], dtype=np.uint8)
        mask = np.zeros(shape[:2], dtype=np.uint8)
        left_line, right_line = sorted(lanes, key=lambda line: line[0])
        margin = self.cfg.lane_mask_margin
        lx1, ly1, lx2, ly2 = left_line
        rx1, ry1, rx2, ry2 = right_line
        points = np.array([[lx1 - margin, ly1], [lx2 - margin, ly2], [rx2 + margin, ry2], [rx1 + margin, ry1]], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        return mask

# --- LaneKeeper Class ---
class LaneKeeper:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.angle_history = deque(maxlen=5)

    @staticmethod
    def _x_at_y(line, y_ref):
        x1, y1, x2, y2 = line
        if y1 == y2: return (x1 + x2) // 2
        return int(x1 + (x2 - x1) * (y_ref - y1) / (y2 - y1))

    def calculate_steering(self, lanes, frame_shape):
        h, w = frame_shape[:2]
        if len(lanes) < 2: return 0.0, None, None
        left, right = sorted(lanes, key=lambda ln: ln[0])
        y_ref = int(h * self.cfg.lane_origin_y_ratio)
        x_left = self._x_at_y(left, y_ref)
        x_right = self._x_at_y(right, y_ref)
        lane_center = (x_left + x_right) // 2
        image_center = w // 2
        deviation = lane_center - image_center
        steering_angle = -deviation * self.cfg.steering_gain * self.cfg.max_angle
        self.angle_history.append(steering_angle)
        return np.mean(self.angle_history), deviation, (lane_center, y_ref)

# --- Helper Functions ---
def estimate_distance(cfg: Config, box_h: int, label: str) -> float:
    if box_h <= 0: return float("inf")
    real_height = cfg.OBJECT_REAL_HEIGHTS.get(label, cfg.DEFAULT_OBJECT_HEIGHT)
    distance = (cfg.focal_length * real_height) / box_h
    return distance

def yolo_worker(cfg, frame_queue, results_queue, stop_event):
    print("[INFO] YOLO worker thread started.")
    model = YOLO(cfg.model_path)
    all_class_names = model.names
    target_class_ids = [k for k, v in all_class_names.items() if v in cfg.TARGET_CLASSES]
    print(f"[INFO] YOLO worker will only detect the following classes: {cfg.TARGET_CLASSES}")

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None: break
            t_start = time.time()
            results = model.predict(
                frame,
                imgsz=cfg.imgsz,
                conf=cfg.conf_thresh,
                verbose=False,
                classes=target_class_ids
            )
            processing_time = time.time() - t_start
            if results_queue.empty():
                results_queue.put((results, processing_time))
        except Empty:
            continue
        except Exception as e:
            print(f"[ERROR] YOLO worker failed: {e}")
            if stop_event.is_set(): break
    print("[INFO] YOLO worker thread stopped.")

# --- Main Application ---
def main(cfg: Config):
    lane_detector = LaneDetector(cfg)
    keeper = LaneKeeper(cfg)
    motor = MotorControl(pin_b=19, pin_c=13)
    
    cap = cv2.VideoCapture(cfg.source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {cfg.source}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.resolution[1])

    frame_queue = Queue(maxsize=1)
    results_queue = Queue(maxsize=1)
    stop_event = threading.Event()
    yolo_thread = threading.Thread(target=yolo_worker, args=(cfg, frame_queue, results_queue, stop_event))
    yolo_thread.start()

    lane_detection_enabled = False
    motor_enable = False
    last_known_boxes = []
    frame_counter = 0
    lane_fps, yolo_fps = 0, 0

    def draw_panel(frame, title, lines, origin, panel_width):
        x, y = origin
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, thickness, line_height = 0.5, 1, 20
        panel_height = (len(lines) + 1) * line_height + 15
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, title, (x + 5, y + 18), font, font_scale, (255, 255, 0), thickness, cv2.LINE_AA)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x + 5, y + (i + 2) * line_height), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    print("[INFO] Press 'e' to toggle lane assist. 'm' to toggle motors. 'q' or ESC to quit.")

    try:
        while True:
            t_total_start = time.time()
            ret, frame = cap.read()
            if not ret: 
                print("[INFO] End of video file reached or camera disconnected.")
                break
            
            display_frame = cv2.resize(frame, cfg.resolution)
            h, w = display_frame.shape[:2]

            frame_counter += 1
            if frame_counter % 3 == 0 and frame_queue.empty():
                frame_queue.put(np.copy(display_frame))

            lanes, steering_angle, deviation, lane_center_coords = [], 0.0, None, None
            lane_mask = np.zeros(display_frame.shape[:2], dtype=np.uint8)

            t_lane_start = time.time()
            if lane_detection_enabled:
                lanes = lane_detector.detect_lanes(display_frame)
                steering_angle, deviation, lane_center_coords = keeper.calculate_steering(lanes, display_frame.shape)
                display_frame = lane_detector.draw_lanes(display_frame, lanes)
                if len(lanes) == 2:
                    lane_mask = lane_detector.get_lane_area_mask(display_frame.shape, lanes)
                    overlay = np.zeros_like(display_frame)
                    cv2.fillPoly(overlay, [np.array([lanes[0][:2], lanes[0][2:], lanes[1][2:], lanes[1][:2]], dtype=np.int32)], (255, 255, 255))
                    display_frame = cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0)
                    
                    y_limit = lane_detector.meters_to_y(cfg.detection_distance_m, h)
                    text = f"Detection Limit ({cfg.detection_distance_m:.0f}m)"
                    
                    left_line, right_line = sorted(lanes, key=lambda line: line[0])
                    x_start = keeper._x_at_y(left_line, y_limit)
                    x_end = keeper._x_at_y(right_line, y_limit)

                    if x_start is not None and x_end is not None:
                        cv2.line(display_frame, (x_start, y_limit), (x_end, y_limit), (255, 255, 0), 2)
                        text_x = (x_start + x_end) // 2 - 70
                        cv2.putText(display_frame, text, (text_x, y_limit - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            lane_time = time.time() - t_lane_start
            if lane_time > 0: lane_fps = 1.0 / lane_time

            try:
                new_results, yolo_time = results_queue.get_nowait()
                if yolo_time > 0: yolo_fps = 1.0 / yolo_time
                temp_boxes = []
                for box in new_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = new_results[0].names[int(box.cls[0])]
                    dist = estimate_distance(cfg, y2 - y1, label)
                    bcx, bcy = (x1 + x2) // 2, y2
                    status, color = "Detected", (200, 200, 200)
                    if lane_detection_enabled:
                        if dist > cfg.detection_distance_m:
                            status, color = "Too Far", (0, 255, 255)
                        elif 0 <= bcy < h and 0 <= bcx < w and lane_mask[bcy, bcx] == 255:
                            status, color = "In Lane", (0, 255, 0)
                        else:
                            status, color = "Out of Lane", (0, 0, 255)
                    temp_boxes.append({"xyxy": (x1, y1, x2, y2), "label": label, "dist": dist, "center": (bcx, bcy), "color": color, "status": status})
                last_known_boxes = temp_boxes
            except Empty:
                pass
                
            for box_data in last_known_boxes:
                (x1, y1, x2, y2) = box_data["xyxy"]
                color, label = box_data["color"], box_data["label"]
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"{label} {box_data['dist']:.1f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            total_fps = 1.0 / (time.time() - t_total_start)
            perf_lines = [f"Total: {total_fps:.1f} FPS", f"Lane : {lane_fps:.1f} FPS" if lane_detection_enabled else "Lane : OFF", f"YOLO : {yolo_fps:.1f} FPS"]
            draw_panel(display_frame, "PERFORMANCE", perf_lines, (w - 170, 10), 160)
            if lane_detection_enabled and motor_enable:
                driving_lines = [f"Rec. Speed : {cfg.normal_speed} km/h", f"Rec. Angle : {steering_angle:.1f}", f"Motor Angle: {motor.current_angle:.1f}"]
                draw_panel(display_frame, "DRIVING INFO", driving_lines, (10, 10), 200)
            else:
                status_lines = [f"Lane Assist: {'ON' if lane_detection_enabled else 'OFF'}", f"Motor      : {'ON' if motor_enable else 'OFF'}"]
                draw_panel(display_frame, "SYSTEM STATUS", status_lines, (10, 10), 200)
            all_labels = [data['label'] for data in last_known_boxes]
            if lane_detection_enabled:
                in_lane_labels = [data['label'] for data in last_known_boxes if data['status'] == 'In Lane']
                out_lane_labels = [data['label'] for data in last_known_boxes if data['status'] == 'Out of Lane']
                too_far_labels = [data['label'] for data in last_known_boxes if data['status'] == 'Too Far']
                in_str = ", ".join([f"{c} {l}" for l, c in Counter(in_lane_labels).items()])
                out_str = ", ".join([f"{c} {l}" for l, c in Counter(out_lane_labels).items()])
                far_str = ", ".join([f"{c} {l}" for l, c in Counter(too_far_labels).items()])
                obj_lines = [f"In Lane : {in_str or 'None'}", f"Out Lane: {out_str or 'None'}", f"Too Far : {far_str or 'None'}"]
            else:
                all_str = ", ".join([f"{c} {l}" for l, c in Counter(all_labels).items()])
                obj_lines = [f"Detected: {all_str or 'None'}"]
            draw_panel(display_frame, "DETECTED OBJECTS", obj_lines, (10, h - (len(obj_lines)*20 + 35)), 250)

            if lane_detection_enabled and lane_center_coords:
                lc_x, lc_y = lane_center_coords
                cv2.line(display_frame, (w // 2, lc_y), (lc_x, lc_y), (0, 0, 255), 3)
                cv2.circle(display_frame, (lc_x, lc_y), 8, (0, 255, 0), -1)
                cv2.putText(display_frame, f"Deviation: {deviation}px", (w // 2 - 50, lc_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if motor_enable:
                motor.move_to(steering_angle)
            else:
                motor.set_stop()

            cv2.imshow("Lane and Object Detection System", display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27: break
            if key == ord('e'):
                lane_detection_enabled = not lane_detection_enabled
                if not lane_detection_enabled: motor_enable = False
            if key == ord('m'):
                if lane_detection_enabled: motor_enable = not motor_enable
                else: motor_enable = False
    finally:
        print("\nStopping threads...")
        stop_event.set()
        if frame_queue.empty(): frame_queue.put(None) 
        yolo_thread.join()
        cap.release()
        cv2.destroyAllWindows()
        motor.stop()
        print("Cleanup complete.")

if __name__ == "__main__":
    config = Config()

    while True:
        print("--- Video Source Selection ---")
        choice = input("  1: Live Camera\n  2: Video File\nEnter your choice (1 or 2): ")
        if choice == '1':
            while True:
                cam_index_str = input("Enter camera index [default: 0]: ")
                if not cam_index_str: config.source = 0; break
                elif cam_index_str.isdigit(): config.source = int(cam_index_str); break
                else: print("[ERROR] Invalid input. Please enter a number.")
            print(f"[INFO] Using camera index: {config.source}")
            break
        elif choice == '2':
            while True:
                video_path = "/home/rpi/Downloads/สร้างวิดีโอกล้องหน้ารถที่กำลัง.mp4"#input("Enter the full path to your video file: ")
                if video_path: config.source = video_path; break
                else: print("[ERROR] Path cannot be empty.")
            print(f"[INFO] Using video file: {config.source}")
            break
        else:
            print("\n[ERROR] Invalid choice. Please enter 1 or 2.\n")
    
    while True:
        prompt = f"\nEnter detection distance in meters [default: {config.detection_distance_m}]: "
        dist_str = input(prompt)
        if not dist_str:
            print(f"[INFO] Using default distance: {config.detection_distance_m}m")
            break
        try:
            dist_val = float(dist_str)
            if dist_val > 0:
                config.detection_distance_m = dist_val
                print(f"[INFO] Set detection distance to: {config.detection_distance_m}m")
                break
            else:
                print("[ERROR] Distance must be a positive number.")
        except ValueError:
            print("[ERROR] Invalid input. Please enter a number (e.g., 70.0).")

    print(f"\n[INFO] Starting detection...")
    main(config)
