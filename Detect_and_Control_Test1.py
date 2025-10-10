#!/usr/st/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
from dataclasses import dataclass
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
    source: int = 0
    resolution: tuple = (640,480)
    conf_thresh: float = 0.2
    imgsz: int = 226
    steering_gain: float = 0.02
    max_angle: float = 30.0
    roi_top_ratio: float = 0.5
    canny_low: int = 50
    canny_high: int = 150
    focal_length: float = 700.0
    normal_speed: int = 30
    # --- [NEW] Maximum detection distance in meters ---
    max_detection_dist_m: float = 25

# --- MotorControl Class (Unchanged) ---
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

# --- LaneDetector and LaneKeeper Classes (Unchanged) ---
class LaneDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.left_lanes = deque(maxlen=10)
        self.right_lanes = deque(maxlen=10)

    def _average_lines(self, lines, height):
        if not lines: return None
        avg_line = np.mean(np.array(lines), axis=0, dtype=np.int32)
        x1, y1, x2, y2 = avg_line
        if x1 == x2: return None
        slope = (y2 - y1) / (x2 - x1)
        if slope == 0: return None
        intercept = y1 - slope * x1
        y1_new = int(height * 0.9)
        y2_new = int(height * 0.65)
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
        points = np.array([left_line[:2], left_line[2:], right_line[2:], right_line[:2]], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        return mask

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
        y_ref = int(h * 0.9)
        x_left = self._x_at_y(left, y_ref)
        x_right = self._x_at_y(right, y_ref)
        lane_center = (x_left + x_right) // 2
        image_center = w // 2
        deviation = lane_center - image_center
        steering_angle = -deviation * self.cfg.steering_gain * self.cfg.max_angle
        self.angle_history.append(steering_angle)
        return np.mean(self.angle_history), deviation, (lane_center, y_ref)

# --- YOLO Worker & Helper Functions (Unchanged) ---
def estimate_distance(cfg: Config, box_h: int) -> float:
    if box_h <= 0: return float("inf")
    return (cfg.focal_length * 1.5) / box_h

def yolo_worker(cfg, frame_queue, results_queue, stop_event):
    print("[INFO] YOLO worker thread started.")
    model = YOLO(cfg.model_path)
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None: break
            
            # --- [MODIFIED] Time the prediction ---
            t_start = time.time()
            results = model.predict(frame, imgsz=cfg.imgsz, conf=cfg.conf_thresh, verbose=False)
            processing_time = time.time() - t_start
            
            if results_queue.empty():
                # --- [MODIFIED] Put both results and time in the queue ---
                results_queue.put((results, processing_time))

        except Empty:
            continue
        except Exception as e:
            print(f"[ERROR] YOLO worker failed: {e}")
            if stop_event.is_set(): break
    print("[INFO] YOLO worker thread stopped.")

# --- Main Application ---
def main(cfg: Config):
    # --- Setup ---
    lane_detector = LaneDetector(cfg)
    keeper = LaneKeeper(cfg)
    motor = MotorControl(pin_b=19, pin_c=13)
    cap = cv2.VideoCapture(cfg.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.resolution[1])
    
    # --- Multithreading Setup ---
    frame_queue = Queue(maxsize=1)
    results_queue = Queue(maxsize=1)
    stop_event = threading.Event()
    yolo_thread = threading.Thread(target=yolo_worker, args=(cfg, frame_queue, results_queue, stop_event))
    yolo_thread.start()

    # --- [MODIFIED] State Variables with new FPS counters ---
    lane_detection_enabled = False
    motor_enable = False
    last_known_boxes = []
    frame_counter = 0
    lane_fps, yolo_fps = 0, 0 # Initialize FPS counters

    # --- [NEW] Helper function for drawing clean text panels ---
    def draw_panel(frame, title, lines, origin, panel_width):
        x, y = origin
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 20
        
        panel_height = (len(lines) + 1) * line_height + 15
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Draw Title
        cv2.putText(frame, title, (x + 5, y + 18), font, font_scale, (255, 255, 0), thickness, cv2.LINE_AA)
        
        # Draw Lines
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x + 5, y + (i + 2) * line_height), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    print("[INFO] Press 'e' to toggle lane assist. 'm' to toggle motors. 'q' or ESC to quit.")

    try:
        while True:
            t_total_start = time.time()
            ret, frame = cap.read()
            if not ret: break
            
            display_frame = cv2.resize(frame, cfg.resolution)
            h, w = display_frame.shape[:2]

            # --- Frame Skipping for YOLO ---
            frame_counter += 1
            if frame_counter % 3 == 0:
                if frame_queue.empty():
                    frame_queue.put(np.copy(display_frame))

            # --- Initialize variables for current frame ---
            lanes, steering_angle, deviation, lane_center_coords = [], 0.0, None, None
            lane_mask = np.zeros(display_frame.shape[:2], dtype=np.uint8)

            # --- Centralized Lane Processing ---
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
                    
                    y_limit = int(h * 0.65)
                    left_line, right_line = sorted(lanes, key=lambda line: line[0])
                    rx1, ry1, rx2, ry2 = right_line
                    midpoint_x = (rx1 + rx2) // 2
                    midpoint_y = (ry1 + ry2) // 2
                    text = f"Distance ({cfg.max_detection_dist_m:.0f}m)"
                    text_position = (midpoint_x - 20, midpoint_y - 20) # 10px to the right of the line
                    cv2.putText(display_frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    x_start = keeper._x_at_y(left_line, y_limit)
                    x_end = keeper._x_at_y(right_line, y_limit)
                    if x_start is not None and x_end is not None:
                        cv2.line(display_frame, (x_start, y_limit), (x_end, y_limit), (255, 255, 0), 2)
                        text_x = (x_start + x_end) // 2 - 70 # Adjust for text width
                        cv2.putText(display_frame, f"Detection Limit", 
                                    (text_x + 10, y_limit + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            lane_time = time.time() - t_lane_start
            if lane_time > 0: lane_fps = 1.0 / lane_time

            # --- Bounding Box Handling ---
            try:
                # [MODIFIED] Unpack results and processing time
                new_results, yolo_time = results_queue.get_nowait()
                if yolo_time > 0: yolo_fps = 1.0 / yolo_time
                
                temp_boxes = []
                for box in new_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = new_results[0].names[int(box.cls[0])]
                    dist = estimate_distance(cfg, y2 - y1)
                    bcx, bcy = (x1 + x2) // 2, y2
                    
                    # ... ภายในลูป for box in new_results[0].boxes: ...

                    status = ""
                    color = (200, 200, 200) # Default to a neutral gray color

                    # Only apply specific colors and status if lane detection is ON
                    if lane_detection_enabled:
                        # 1. ตรวจสอบก่อนเลยว่าวัตถุอยู่ไกลเกินไปหรือไม่ (สำคัญที่สุด)
                        if dist > cfg.max_detection_dist_m:
                            status, color = "Too Far", (0, 255, 255)      # Yellow

                        # 2. ถ้าไม่ไกลเกินไป ค่อยเช็คว่าอยู่ในเลนหรือไม่
                        elif 0 <= bcy < h and 0 <= bcx < w and lane_mask[bcy, bcx] == 255:
                            status, color = "In Lane", (0, 255, 0)      # Green

                        # 3. กรณีสุดท้ายคืออยู่นอกเลน (แต่ยังอยู่ในระยะ)
                        else:
                            status, color = "Out of Lane", (0, 0, 255)  # Red
                    else:
                        # If lane detection is OFF, all objects are simply "Detected"
                        status = "Detected"

                    temp_boxes.append({
                        "xyxy": (x1, y1, x2, y2), "label": label, "dist": dist,
                        "center": (bcx, bcy), "color": color, "status": status
                    })
                
                # [FIXED] Corrected the typo in the variable name
                last_known_boxes = temp_boxes 
                
            except Empty:
                pass

            # --- Drawing Boxes ---
            in_lane_labels, out_lane_labels, too_far_labels = [], [], []
            if last_known_boxes:
                for box_data in last_known_boxes:
                    (x1, y1, x2, y2) = box_data["xyxy"]
                    color, label = box_data["color"], box_data["label"]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, f"{label} {box_data['dist']:.1f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    if box_data["status"] == "In Lane": in_lane_labels.append(label)
                    elif box_data["status"] == "Out of Lane": out_lane_labels.append(label)
                    else: too_far_labels.append(label)

            # --- [NEW] Organized Info Display ---
            total_fps = 1.0 / (time.time() - t_total_start)

            # Panel 1: Performance (Top Right)
            perf_lines = [
                f"Total: {total_fps:.1f} FPS",
                f"Lane : {lane_fps:.1f} FPS" if lane_detection_enabled else "Lane : OFF",
                f"YOLO : {yolo_fps:.1f} FPS"
            ]
            draw_panel(display_frame, "PERFORMANCE", perf_lines, (w - 170, 10), 160)
            
            # Panel 2: System Status (Top Left)
            # Panel for System Status / Driving Info (Top Left)
            # This panel conditionally switches between two displays.

            if lane_detection_enabled and motor_enable:
                # STATE 1: Show "DRIVING INFO" when both systems are active.
                driving_lines = [
                    f"Rec. Speed : {cfg.normal_speed} km/h",
                    f"Rec. Angle : {steering_angle:.1f}",
                    f"Motor Angle: {motor.current_angle:.1f}"
                ]
                draw_panel(display_frame, "DRIVING INFO", driving_lines, (10, 10), 200)

            else:
                # STATE 2: Show "SYSTEM STATUS" if either system is off.
                status_lines = [
                    f"Lane Assist: {'ON' if lane_detection_enabled else 'OFF'}",
                    f"Motor      : {'ON' if motor_enable else 'OFF'}"
                ]
                draw_panel(display_frame, "SYSTEM STATUS", status_lines, (10, 10), 200)
            
            # Panel 3: Objects (Bottom Left)
            # First, collect all object labels from the current frame
            all_labels = [data['label'] for data in last_known_boxes]

            if lane_detection_enabled:
                # If lane assist is ON, create categorized lists
                in_lane_labels = [data['label'] for data in last_known_boxes if data['status'] == 'In Lane']
                out_lane_labels = [data['label'] for data in last_known_boxes if data['status'] == 'Out of Lane']
                too_far_labels = [data['label'] for data in last_known_boxes if data['status'] == 'Too Far']
                
                # --- [FIXED] Changed l[0] to l to show full label names ---
                in_str = ", ".join([f"{c} {l}" for l, c in Counter(in_lane_labels).items()])
                out_str = ", ".join([f"{c} {l}" for l, c in Counter(out_lane_labels).items()])
                far_str = ", ".join([f"{c} {l}" for l, c in Counter(too_far_labels).items()])
                
                obj_lines = [
                    f"In Lane : {in_str or 'None'}",
                    f"Out Lane: {out_str or 'None'}",
                    f"Too Far : {far_str or 'None'}"
                ]
            else:
                # If lane assist is OFF, create one combined list
                # --- [FIXED] Changed l[0] to l here as well ---
                all_str = ", ".join([f"{c} {l}" for l, c in Counter(all_labels).items()])
                obj_lines = [
                    f"Detected: {all_str or 'None'}"
                ]

            # --- [MODIFIED] Increased panel width to accommodate longer text ---
            draw_panel(display_frame, "DETECTED OBJECTS", obj_lines, (230, 10), 220)
            
            # --- Visualizations & Motor Control ---
            if lane_detection_enabled and lane_center_coords:
                lc_x, lc_y = lane_center_coords
                cv2.line(display_frame, (w // 2, lc_y), (lc_x, lc_y), (0, 0, 255), 3)
                cv2.circle(display_frame, (lc_x, lc_y), 8, (0, 255, 0), -1)
                cv2.putText(display_frame, f"Deviation: {deviation}px", (w // 2 - 50, lc_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if motor_enable:
                motor.move_to(steering_angle)
            else:
                motor.set_stop()

            cv2.imshow("High-Performance Lane and Object Detection", display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27: break
            if key == ord('e'):
                lane_detection_enabled = not lane_detection_enabled
                if not lane_detection_enabled:
                    motor_enable = False
            if key == ord('m'):
                if lane_detection_enabled:
                    motor_enable = not motor_enable
                else:
                    motor_enable = False
    finally:
        # --- Graceful Shutdown ---
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
    main(config)
