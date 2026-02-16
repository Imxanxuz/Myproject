#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time 
from time import sleep
from dataclasses import dataclass, field
from ultralytics import YOLO
import RPi.GPIO as GPIO
from collections import deque, Counter
import threading
from queue import Queue, Empty

GPIO.setwarnings(False)
GPIO_MODE = True

@dataclass
class Config:
    model_path: str = "/home/rpi/yolo/yolo11n.pt"
    source: any = 0
    resolution: tuple = (480, 320)
    conf_thresh: float = 0.25  
    imgsz: int = 192
    
    # --- Steering & PID Control ---
    STEERING_RANGE: tuple = (60.0, 120.0) 
    STEERING_CENTER: float = 90.0
    PID_GAINS: tuple = (0.007, 0.015, 0.005) 
    PID_WINDUP_LIMIT: float = 100.0  

    # --- Lane Detection Tuning ---
    roi_top_ratio: float = 0.65
    roi_top_left_x_ratio: float = 0.40
    roi_top_right_x_ratio: float = 0.60
    canny_low: int = 40
    canny_high: int = 120
    min_lane_slope: float = 0.3
    poly_fit_deque_len: int = 6
    poly_fit_margin: int = 50
    poly_min_points_for_fit: int = 15
    
    LANE_SANITY_CHECK_PX: tuple = (200, 750)
    LANE_SANITY_CHECK_RATIO: tuple = (0.6, 3.5)
    roi_bottom_left_x_ratio: float = 0.05 
    roi_bottom_right_x_ratio: float = 0.95 

    # --- Obstacle Avoidance ---
    safe_distance: float = 2.5 
    critical_stop: float = 1.2 
    side_check_width: int = 100 
    lane_mask_margin: int = 50
    focal_length: float = 400.0
    lane_origin_y_ratio: float = 0.80 
    TARGET_CLASSES: list = field(default_factory=lambda: ["person", "car", "bus", "bicycle", "motorcycle", "dog"])
    OBJECT_REAL_HEIGHTS: dict = field(default_factory=lambda: {
        "person": 1.7, "car": 1.5, "truck": 3.5, "bus": 3.2, "bicycle": 1.0, "motorcycle": 1.2, "dog": 0.5
    })
    DEFAULT_OBJECT_HEIGHT: float = 1.0

class MotorControl_to_MotorDriver:
    def __init__(self, RPWM=13, LPWM=12, R_EN=17, L_EN=27, S1=5, S2=6, freq=1000):
        # BTS7960 Pins
        self.RPWM = RPWM  # Forward PWM
        self.LPWM = LPWM  # Reverse PWM
        self.R_EN = R_EN  # Forward Enable
        self.L_EN = L_EN  # Reverse Enable
        
        # Relay Pins for Steering (?? P-215)
        self.S1, self.S2 = S1, S2 
        
        self.current_speed = 0
        self.center_angle = 90.0
        
        if GPIO_MODE:
            GPIO.setmode(GPIO.BCM)
            for pin in [self.RPWM, self.LPWM, self.R_EN, self.L_EN, self.S1, self.S2]:
                GPIO.setup(pin, GPIO.OUT)
            

            GPIO.output(self.R_EN, GPIO.HIGH)
            GPIO.output(self.L_EN, GPIO.HIGH)
            

            self.pwm_forward = GPIO.PWM(self.RPWM, freq)
            self.pwm_reverse = GPIO.PWM(self.LPWM, freq)
            self.pwm_forward.start(0)
            self.pwm_reverse.start(0)

    def steer_left(self):
        if GPIO_MODE:
            GPIO.output(self.S1, GPIO.HIGH)
            GPIO.output(self.S2, GPIO.LOW)

    def steer_right(self):
        if GPIO_MODE:
            GPIO.output(self.S1, GPIO.LOW)
            GPIO.output(self.S2, GPIO.HIGH)

    def steer_straight(self):
        if GPIO_MODE:
            GPIO.output(self.S1, GPIO.LOW)
            GPIO.output(self.S2, GPIO.LOW)

    def steer(self, angle):
        dev = angle - self.center_angle
        if dev < -5:
            self.steer_left()
        elif dev > 5:
            self.steer_right()
        else:
            self.steer_straight()

    def drive(self, speed):
        """ ???????????????????????? (???????? 0-100) """
        self.current_speed = speed
        if GPIO_MODE:
            # ?????? BTS7960 ????????: ??? PWM ???? RPWM, LPWM ???????? 0
            self.pwm_reverse.ChangeDutyCycle(0)
            self.pwm_forward.ChangeDutyCycle(speed)

    def drive_reverse(self, speed):
        """ ??????????? (???????? 0-100) """
        self.current_speed = speed
        if GPIO_MODE:
            # ?????? BTS7960 ???????: ??? PWM ???? LPWM, RPWM ???????? 0
            self.pwm_forward.ChangeDutyCycle(0)
            self.pwm_reverse.ChangeDutyCycle(speed)

    def move_to(self, angle, base_speed=50): # ???? base_speed ??????????????? BTS7960 ???????????
        self.steer(angle)
        self.drive(base_speed)

    def set_stop(self):
        if GPIO_MODE:
            self.pwm_forward.ChangeDutyCycle(0)
            self.pwm_reverse.ChangeDutyCycle(0)
            self.steer_straight()

    def stop(self):
        if GPIO_MODE:
            self.pwm_forward.stop()
            self.pwm_reverse.stop()
            GPIO.cleanup()
class Safety:
    def __init__(self, move_stop=2.0, move_to=5.0):
        self.move_stop = move_stop
        self.move_to = move_to
        self.auto_stop_active = False 

    def movement_update(self, last_known_boxes):
        # Filter only objects that are "In Lane"
        in_lane_objects = [data for data in last_known_boxes if data['status'] == 'In Lane']
        
        if in_lane_objects:
            # Extract distances of all objects in the lane
            distances = [data['dist'] for data in in_lane_objects]
            min_dist = min(distances) 
            
            # Logic: If closest object is nearer than move_stop, activate brake
            if min_dist <= self.move_stop:
                self.auto_stop_active = True
            else:
                self.auto_stop_active = False
            
            return self.auto_stop_active, min_dist
        else:
            self.auto_stop_active = False 
            return False, None
class LaneDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.left_fit_history = deque(maxlen=cfg.poly_fit_deque_len)
        self.right_fit_history = deque(maxlen=cfg.poly_fit_deque_len)
        self.current_left_fit = None
        self.current_right_fit = None
        self.last_good_fits = (None, None) 
        self.last_good_width_px = (cfg.LANE_SANITY_CHECK_PX[0] + cfg.LANE_SANITY_CHECK_PX[1]) / 2

    def meters_to_y(self, meters: float, height: int) -> int:

        m_near, m_far = self.cfg.LANE_DIST_CALIB_M
        y_near, y_far = height, int(height * self.cfg.roi_top_ratio)
        y_coord = np.interp(meters, [m_near, m_far], [y_near, y_far])
        return int(np.clip(y_coord, y_far, y_near))

    def _get_x_at_y(self, fit, y_ref):
        if fit is None: return None
        return fit[0] * y_ref**2 + fit[1] * y_ref + fit[2]

    def detect_lanes(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.cfg.canny_low, self.cfg.canny_high)
        h, w = edges.shape
        mask = np.zeros_like(edges)
        
        pts = np.array([
            [(int(w * self.cfg.roi_bottom_left_x_ratio), h), 
             (int(w * self.cfg.roi_top_left_x_ratio), int(h * self.cfg.roi_top_ratio)), 
             (int(w * self.cfg.roi_top_right_x_ratio), int(h * self.cfg.roi_top_ratio)), 
             (int(w * self.cfg.roi_bottom_right_x_ratio), h)]
        ], np.int32)
        
        cv2.fillPoly(mask, pts, 255)
        roi = cv2.bitwise_and(edges, mask)
        
        lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 20, minLineLength=20, maxLineGap=40)
        
        left_x, left_y = [], []
        right_x, right_y = [], []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1: continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < self.cfg.min_lane_slope: continue
                
                if slope < 0:
                    left_x.extend([x1, x2])
                    left_y.extend([y1, y2])
                else:
                    right_x.extend([x1, x2])
                    right_y.extend([y1, y2])

        if len(left_y) > self.cfg.poly_min_points_for_fit:
            current_y = np.array(left_y)
            current_x = np.array(left_x)
            if self.left_fit_history:
                prev_fit = np.mean(self.left_fit_history, axis=0)
                predicted_x = self._get_x_at_y(prev_fit, current_y)
                errors = np.abs(current_x - predicted_x)
                keep_indices = errors < self.cfg.poly_fit_margin
                filtered_y = current_y[keep_indices]
                filtered_x = current_x[keep_indices]
                
                if len(filtered_y) > self.cfg.poly_min_points_for_fit:
                    left_fit = np.polyfit(filtered_y, filtered_x, 2)
                    self.left_fit_history.append(left_fit)
            elif len(current_y) > self.cfg.poly_min_points_for_fit: 
                left_fit = np.polyfit(current_y, current_x, 2)
                self.left_fit_history.append(left_fit)

        if len(right_y) > self.cfg.poly_min_points_for_fit:
            current_y = np.array(right_y)
            current_x = np.array(right_x)
            if self.right_fit_history:
                prev_fit = np.mean(self.right_fit_history, axis=0)
                predicted_x = self._get_x_at_y(prev_fit, current_y)
                errors = np.abs(current_x - predicted_x)
                keep_indices = errors < self.cfg.poly_fit_margin
                filtered_y = current_y[keep_indices]
                filtered_x = current_x[keep_indices]
                
                if len(filtered_y) > self.cfg.poly_min_points_for_fit:
                    right_fit = np.polyfit(filtered_y, filtered_x, 2)
                    self.right_fit_history.append(right_fit)
            elif len(current_y) > self.cfg.poly_min_points_for_fit: 
                right_fit = np.polyfit(current_y, current_x, 2)
                self.right_fit_history.append(right_fit)

        if self.left_fit_history:
            self.current_left_fit = np.mean(self.left_fit_history, axis=0)
        if self.right_fit_history:
            self.current_right_fit = np.mean(self.right_fit_history, axis=0)

        is_sane = False 
        if self.current_left_fit is not None and self.current_right_fit is not None:
            y_bottom = h - 1
            y_top = int(h * self.cfg.roi_top_ratio)
            
            x_left_bottom = self._get_x_at_y(self.current_left_fit, y_bottom)
            x_right_bottom = self._get_x_at_y(self.current_right_fit, y_bottom)
            x_left_top = self._get_x_at_y(self.current_left_fit, y_top)
            x_right_top = self._get_x_at_y(self.current_right_fit, y_top)

            if x_left_bottom is None or x_right_bottom is None or x_left_top is None or x_right_top is None:
                self.current_left_fit, self.current_right_fit = self.last_good_fits
                return self.current_left_fit, self.current_right_fit

            width_bottom = x_right_bottom - x_left_bottom
            width_top = x_right_top - x_left_top

            min_px, max_px = self.cfg.LANE_SANITY_CHECK_PX
            min_r, max_r = self.cfg.LANE_SANITY_CHECK_RATIO
            
            check1 = (min_px < width_bottom < max_px)
            check2 = (width_top > 0 and (min_r < (width_bottom / width_top) < max_r))
            check3 = (abs(width_bottom - self.last_good_width_px) <= 150) 

            if check1 and check2 and check3:
                is_sane = True
                self.last_good_fits = (self.current_left_fit, self.current_right_fit)
                self.last_good_width_px = width_bottom
        
        if not is_sane and self.last_good_fits[0] is not None:
            self.current_left_fit, self.current_right_fit = self.last_good_fits
        elif (self.current_left_fit is None or self.current_right_fit is None) and self.last_good_fits[0] is not None:
            self.current_left_fit, self.current_right_fit = self.last_good_fits

        return self.current_left_fit, self.current_right_fit

    def draw_lanes(self, frame, left_fit, right_fit):
        if left_fit is None and right_fit is None:
            return frame
        lane_img = np.zeros_like(frame)
        h, w = frame.shape[:2]
        plot_y = np.linspace(int(h * self.cfg.roi_top_ratio), h - 1, int(h * (1 - self.cfg.roi_top_ratio)))
        try:
            if left_fit is not None:
                left_fit_x = self._get_x_at_y(left_fit, plot_y)
                left_points = np.asarray([left_fit_x, plot_y]).T.astype(np.int32)
                cv2.polylines(lane_img, [left_points], isClosed=False, color=(0, 255, 255), thickness=4)
            if right_fit is not None:
                right_fit_x = self._get_x_at_y(right_fit, plot_y)
                right_points = np.asarray([right_fit_x, plot_y]).T.astype(np.int32)
                cv2.polylines(lane_img, [right_points], isClosed=False, color=(0, 255, 255), thickness=4)
        except Exception as e:
            print(f"[WARN] Error drawing polylines: {e}")
        return cv2.addWeighted(frame, 1.0, lane_img, 1.0, 0)

    def get_lane_area_mask(self, shape, left_fit, right_fit):
        if left_fit is None or right_fit is None:
            return np.zeros(shape[:2], dtype=np.uint8)
        h, w = shape[:2]
        mask = np.zeros(shape[:2], dtype=np.uint8)
        margin = self.cfg.lane_mask_margin
        plot_y = np.linspace(int(h * self.cfg.roi_top_ratio), h - 1, 20)
        left_fit_x = self._get_x_at_y(left_fit, plot_y)
        right_fit_x = self._get_x_at_y(right_fit, plot_y)
        
        if left_fit_x is None or right_fit_x is None:
            return np.zeros(shape[:2], dtype=np.uint8)
            
        pts_left = np.asarray([left_fit_x - margin, plot_y]).T
        pts_right = np.asarray([right_fit_x + margin, plot_y]).T
        points = np.vstack([pts_left, np.flipud(pts_right)]).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
        return mask
class LaneKeeper:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.kp, self.ki, self.kd = cfg.PID_GAINS
        self.windup_limit = cfg.PID_WINDUP_LIMIT
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def _x_at_y(self, fit, y_ref):
        if fit is None: return None
        return fit[0] * y_ref**2 + fit[1] * y_ref + fit[2]

    def calculate_steering(self, fits, frame_shape):
        left_fit, right_fit = fits
        h, w = frame_shape[:2]
        current_time = time.time()
        dt = current_time - self.last_time
        if dt == 0: dt = 1e-5 
        self.last_time = current_time
        
        if left_fit is None or right_fit is None:
            self.last_error = 0.0
            self.integral = 0.0
            return self.cfg.STEERING_CENTER, None, None
            
        y_ref = int(h * self.cfg.lane_origin_y_ratio) 
        
        x_left = self._x_at_y(left_fit, y_ref)
        x_right = self._x_at_y(right_fit, y_ref)
        
        if x_left is None or x_right is None:
            self.last_error = 0.0
            self.integral = 0.0
            return self.cfg.STEERING_CENTER, None, None
            
        lane_center = (x_left + x_right) // 2
        image_center = w // 2
        deviation = lane_center - image_center 
        
        error = -deviation  
        
        P = self.kp * error
        
        self.integral = self.integral + (error * dt) 
        self.integral = np.clip(self.integral, -self.windup_limit, self.windup_limit)
        I = self.ki * self.integral
        
        derivative = (error - self.last_error) / dt
        D = self.kd * derivative
        
        self.last_error = error
        
        pid_output = np.clip(P + I + D, -1.0, 1.0)
        
        final_angle = np.interp(
            pid_output,
            [-1.0, 0.0, 1.0],
            [self.cfg.STEERING_RANGE[0], self.cfg.STEERING_CENTER, self.cfg.STEERING_RANGE[1]]
        )
        
        return final_angle, deviation, (lane_center, y_ref)
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

def main(cfg: Config):
    lane_detector = LaneDetector(cfg)
    keeper = LaneKeeper(cfg)
    # à¹ƒà¸Šà¹‰à¸‚à¸² ENA, ENB, S1, S2 à¸•à¸²à¸¡à¸—à¸µà¹ˆà¸à¸³à¸«à¸™à¸”
    motor = MotorControl_to_MotorDriver(RPWM=13, LPWM=12, R_EN=17, L_EN=27, S1=5, S2=6)
    auto_stop = Safety(move_stop=2.0, move_to=5.0)
    
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

    # --- à¸Šà¸·à¹ˆà¸­ Parameter à¸•à¸²à¸¡à¸•à¹‰à¸™à¸‰à¸šà¸±à¸š ---
    lane_detection_enabled = False  # à¸„à¸§à¸šà¸„à¸¸à¸¡à¸”à¹‰à¸§à¸¢à¸›à¸¸à¹ˆà¸¡ 'L'
    obj_detect_enabled = False      # à¸„à¸§à¸šà¸„à¸¸à¸¡à¸”à¹‰à¸§à¸¢à¸›à¸¸à¹ˆà¸¡ 'e' (à¹€à¸žà¸´à¹ˆà¸¡à¸¡à¸²à¹ƒà¸«à¸¡à¹ˆà¹€à¸žà¸·à¹ˆà¸­à¹à¸¢à¸à¸à¸±à¸™)
    motor_enable = False           # à¸„à¸§à¸šà¸„à¸¸à¸¡à¸”à¹‰à¸§à¸¢à¸›à¸¸à¹ˆà¸¡ 'm'
    
    last_known_boxes = []
    frame_counter = 0
    lane_fps, yolo_fps, total_fps = 0, 0, 0
    last_motor_update_time = 0.0

    print("[INFO] 'L': Lane, 'e': Object, 'm': Motor, 'q': Quit")

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

            # --- 1. YOLO Processing Logic (à¸„à¸§à¸šà¸„à¸¸à¸¡à¸”à¹‰à¸§à¸¢à¸›à¸¸à¹ˆà¸¡ 'e') ---
            frame_counter += 1
            if obj_detect_enabled: # à¹€à¸Šà¹‡à¸„à¸ªà¸–à¸²à¸™à¸°à¸à¸²à¸£à¹€à¸›à¸´à¸”à¸•à¸£à¸§à¸ˆà¸ˆà¸±à¸šà¸§à¸±à¸•à¸–à¸¸
                if frame_counter % 3 == 0 and frame_queue.empty():
                    frame_queue.put(np.copy(display_frame))
            
            try:
                # à¸”à¸¶à¸‡à¸œà¸¥à¸¥à¸±à¸žà¸˜à¹Œà¸ˆà¸²à¸ YOLO Thread
                new_results, yolo_time = results_queue.get_nowait()
                if yolo_time > 0: yolo_fps = 1.0 / yolo_time
                temp_boxes = []
                for box in new_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = new_results[0].names[int(box.cls[0])]
                    dist = estimate_distance(cfg, y2 - y1, label)
                    bcx, bcy = (x1 + x2) // 2, y2
                    
                    status = "In Lane" if (0 <= bcy < h and 0 <= bcx < w and lane_mask[bcy, bcx] == 255) else "Side"
                    
                    temp_boxes.append({
                        "xyxy": (x1, y1, x2, y2), 
                        "label": label, 
                        "dist": dist, 
                        "center": (bcx, bcy), 
                        "cx": bcx, # เพิ่มค่านี้เพื่อให้ Logic หลบหลีกเรียกใช้ได้
                        "color": (0, 255, 0) if status == "In Lane" else (255, 255, 0), 
                        "status": status
                    })
                last_known_boxes = temp_boxes
            except Empty:
                pass

            # à¸§à¸²à¸”à¸à¸£à¸­à¸šà¸§à¸±à¸•à¸–à¸¸à¸–à¹‰à¸²à¹€à¸›à¸´à¸”à¹‚à¸«à¸¡à¸”à¹„à¸§à¹‰
            if obj_detect_enabled:
                for box_data in last_known_boxes:
                    (x1, y1, x2, y2) = box_data["xyxy"]
                    color, label, d = box_data["color"], box_data["label"], box_data["dist"]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, f"{label} {d:.1f}m", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- 2. Lane Processing Logic (à¸„à¸§à¸šà¸„à¸¸à¸¡à¸”à¹‰à¸§à¸¢à¸›à¸¸à¹ˆà¸¡ 'L') ---
            left_fit, right_fit = None, None
            # à¸¥à¹‰à¸²à¸‡à¸„à¹ˆà¸² mask à¸—à¸¸à¸à¸£à¸­à¸šà¹€à¸žà¸·à¹ˆà¸­à¸›à¹‰à¸­à¸‡à¸à¸±à¸™à¸„à¸§à¸²à¸¡à¸œà¸´à¸”à¸žà¸¥à¸²à¸”
            lane_mask = np.zeros(display_frame.shape[:2], dtype=np.uint8)

            t_lane_start = time.time()
            # à¸—à¸³à¸à¸²à¸£à¸„à¸³à¸™à¸§à¸“à¹€à¸¥à¸™à¹€à¸ªà¸¡à¸­à¹€à¸žà¸·à¹ˆà¸­à¹ƒà¸Šà¹‰à¹ƒà¸™à¸£à¸°à¸šà¸šà¸«à¸¥à¸šà¸«à¸¥à¸µà¸ (à¹à¸•à¹ˆà¸ˆà¸°à¹à¸ªà¸”à¸‡à¸œà¸¥à¹à¸¥à¸°à¹€à¸à¹‡à¸šà¸›à¸£à¸°à¸§à¸±à¸•à¸´à¹€à¸¡à¸·à¹ˆà¸­à¹€à¸›à¸´à¸”à¸›à¸¸à¹ˆà¸¡ L)
            left_fit, right_fit = lane_detector.detect_lanes(display_frame)
            lane_time = time.time() - t_lane_start
            if lane_time > 0: lane_fps = 1.0 / lane_time

            if left_fit is not None and right_fit is not None:
                lane_mask = lane_detector.get_lane_area_mask(display_frame.shape, left_fit, right_fit)
                
                # à¹à¸ªà¸”à¸‡à¸œà¸¥à¹€à¸ªà¹‰à¸™à¹€à¸¥à¸™à¹€à¸‰à¸žà¸²à¸°à¹€à¸¡à¸·à¹ˆà¸­à¸à¸”à¸›à¸¸à¹ˆà¸¡ L
                if lane_detection_enabled:
                    display_frame = lane_detector.draw_lanes(display_frame, left_fit, right_fit)
                    # à¸ªà¸£à¹‰à¸²à¸‡ Overlay à¸žà¸·à¹‰à¸™à¸—à¸µà¹ˆà¹ƒà¸™à¹€à¸¥à¸™à¹ƒà¸«à¹‰à¸”à¸¹à¹‚à¸›à¸£à¹ˆà¸‡à¹à¸ªà¸‡
                    overlay = np.zeros_like(display_frame)
                    plot_y = np.linspace(int(h * cfg.roi_top_ratio), h - 1, 20)
                    left_fit_x = lane_detector._get_x_at_y(left_fit, plot_y)
                    right_fit_x = lane_detector._get_x_at_y(right_fit, plot_y)
                    if left_fit_x is not None and right_fit_x is not None:
                        pts_left = np.asarray([left_fit_x, plot_y]).T
                        pts_right = np.asarray([right_fit_x, plot_y]).T
                        points = np.vstack([pts_left, np.flipud(pts_right)]).astype(np.int32)
                        cv2.fillPoly(overlay, [points], (0, 255, 100)) # à¸ªà¸µà¹€à¸‚à¸µà¸¢à¸§à¸ˆà¸²à¸‡à¹†
                        display_frame = cv2.addWeighted(overlay, 0.15, display_frame, 0.85, 0)

            # --- 3. Driving Logic & Smart Avoidance ---
            steering_angle = cfg.STEERING_CENTER
            is_blocked, current_min_dist = auto_stop.movement_update(last_known_boxes)
            
            if motor_enable:
                # à¹€à¸‡à¸·à¹ˆà¸­à¸™à¹„à¸‚à¸à¸²à¸£à¸«à¸¥à¸šà¸«à¸¥à¸µà¸ (à¹€à¸Šà¹‡à¸à¸£à¸°à¸¢à¸°à¹€à¸šà¸µà¹ˆà¸¢à¸‡à¸«à¸¥à¸šà¸—à¸µà¹ˆà¸ªà¸¡à¸ˆà¸£à¸´à¸‡)
                if is_blocked and obj_detect_enabled:
                    in_lane_objs = [b for b in last_known_boxes if b['status'] == 'In Lane']
                    if in_lane_objs:
                        target = min(in_lane_objs, key=lambda x: x['dist'])
                        obj_x = target['center'][0]
                        img_center = w // 2
                        
                        # --- à¸•à¸£à¸§à¸ˆà¸ªà¸­à¸šà¹€à¸ªà¹‰à¸™à¸—à¸²à¸‡à¸—à¸µà¹ˆà¸ˆà¸°à¹€à¸šà¸µà¹ˆà¸¢à¸‡à¸«à¸¥à¸š (Side Check) ---
                        if obj_x < img_center: # à¸§à¸±à¸•à¸–à¸¸à¸‚à¸§à¸²à¸‡à¸­à¸¢à¸¹à¹ˆà¸‹à¹‰à¸²à¸¢ -> à¸•à¹‰à¸­à¸‡à¸«à¸¥à¸šà¸‚à¸§à¸²
                            # à¹€à¸Šà¹‡à¸à¸žà¸·à¹‰à¸™à¸—à¸µà¹ˆà¸”à¹‰à¸²à¸™à¸‚à¸§à¸²à¸§à¹ˆà¸²à¸¡à¸µà¸§à¸±à¸•à¸–à¸¸à¸›à¸£à¸°à¸Šà¸´à¸”à¸­à¸¢à¸¹à¹ˆà¹„à¸«à¸¡
                            is_right_clear = not any(b for b in last_known_boxes if b['center'][0] > img_center + 50 and b['dist'] < 2.5)
                            if is_right_clear:
                                motor.steer_right()
                                print(f"[é¿] Avoiding Obstacle: Right Turn (Dist: {target['dist']:.1f}m)")
                            else:
                                motor.set_stop() # à¸–à¹‰à¸²à¹„à¸¡à¹ˆà¸›à¸¥à¸­à¸”à¸ à¸±à¸¢à¸—à¸±à¹‰à¸‡à¸ªà¸­à¸‡à¸à¸±à¹ˆà¸‡à¹ƒà¸«à¹‰à¸«à¸¢à¸¸à¸”
                                print("[!] Critical: Path Blocked on both sides!")
                        else: # à¸§à¸±à¸•à¸–à¸¸à¸‚à¸§à¸²à¸‡à¸­à¸¢à¸¹à¹ˆà¸‚à¸§à¸² -> à¸•à¹‰à¸­à¸‡à¸«à¸¥à¸šà¸‹à¹‰à¸²à¸¢
                            is_left_clear = not any(b for b in last_known_boxes if b['center'][0] < img_center - 50 and b['dist'] < 2.5)
                            if is_left_clear:
                                motor.steer_left()
                                print(f"[é¿] Avoiding Obstacle: Left Turn (Dist: {target['dist']:.1f}m)")
                            else:
                                motor.set_stop()
                                print("[!] Critical: Path Blocked on both sides!")
                        
                        # à¸¥à¸”à¸„à¸§à¸²à¸¡à¹€à¸£à¹‡à¸§à¸‚à¸“à¸°à¹€à¸¥à¸µà¹‰à¸¢à¸§à¸«à¸¥à¸šà¹€à¸žà¸·à¹ˆà¸­à¸„à¸§à¸²à¸¡à¸›à¸¥à¸­à¸”à¸ à¸±à¸¢
                        motor.pwm_left.ChangeDutyCycle(35)
                        motor.pwm_right.ChangeDutyCycle(35)
                
                # à¸à¸£à¸“à¸µà¸›à¸à¸•à¸´: à¸§à¸´à¹ˆà¸‡à¸•à¸²à¸¡à¹€à¸¥à¸™ (à¸–à¹‰à¸²à¹€à¸›à¸´à¸” L) à¸«à¸£à¸·à¸­à¸§à¸´à¹ˆà¸‡à¸•à¸£à¸‡ (à¸–à¹‰à¸²à¹„à¸¡à¹ˆà¹„à¸”à¹‰à¹€à¸›à¸´à¸” L)
                else:
                    if lane_detection_enabled and left_fit is not None and right_fit is not None:
                        steering_angle, _, _ = keeper.calculate_steering((left_fit, right_fit), display_frame.shape)
                        motor.move_to(steering_angle)
                    else:
                        # à¸§à¸´à¹ˆà¸‡à¸•à¸£à¸‡à¹à¸šà¸šà¸›à¸£à¸°à¸„à¸­à¸‡à¸—à¸´à¸¨à¸—à¸²à¸‡
                        motor.steer_straight()
                        motor.drive(60)
            else:
                motor.set_stop()

            # --- 4. Performance & Status Panel ---
            loop_time = time.time() - t_total_start
            total_fps = 1.0 / loop_time if loop_time > 0 else 0
            
            perf_lines = [f"Total: {total_fps:.1f} FPS", f"Yaw  : {lane_fps:.1f} FPS", f"YOLO : {yolo_fps:.1f} FPS"]
            draw_panel(display_frame, "PERFORMANCE", perf_lines, (w - 170, 10), 160)
            
            status_lines = [
                f"Lane Assist: {'ON' if lane_detection_enabled else 'OFF'}", 
                f"Object Det : {'ON' if obj_detect_enabled else 'OFF'}",
                f"Motor      : {'ON' if motor_enable else 'OFF'}", 
                f"Safety     : {'!! STOP !!' if is_blocked else 'NORMAL'}"
            ]
            draw_panel(display_frame, "SYSTEM STATUS", status_lines, (10, 10), 200)

            # --- 5. User Interaction ---
            cv2.imshow("Lane and Object Detection System", display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27: break
            if key == ord('l'): # à¸›à¸¸à¹ˆà¸¡ L à¸„à¸§à¸šà¸„à¸¸à¸¡à¹€à¸¥à¸™
                lane_detection_enabled = not lane_detection_enabled
                print(f"[UI] Lane Detection: {lane_detection_enabled}")
            if key == ord('e'): # à¸›à¸¸à¹ˆà¸¡ e à¸„à¸§à¸šà¸„à¸¸à¸¡à¸à¸²à¸£à¸«à¸²à¸§à¸±à¸•à¸–à¸¸
                obj_detect_enabled = not obj_detect_enabled
                print(f"[UI] Object Detection: {obj_detect_enabled}")
            if key == ord('m'): # à¸›à¸¸à¹ˆà¸¡ m à¸„à¸§à¸šà¸„à¸¸à¸¡à¸¡à¸­à¹€à¸•à¸­à¸£à¹Œ
                motor_enable = not motor_enable
                print(f"[UI] Motor Master: {motor_enable}")
    finally:
        print("\nStopping")
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
        print("Source Selection")
        choice = input("  1: Live Camera\n  2: Video File: ")
        if choice == '1':
            while True:
                cam_index_str = 0
                if not cam_index_str: config.source = 0; break
                elif cam_index_str.isdigit(): config.source = int(cam_index_str); break
                else: print("[ERROR] Invalid input. Please enter a number.")
            print(f"[INFO] Using camera index: {config.source}")
            break
        elif choice == '2':
            while True:
                video_path = "/home/rpi/yolo/video_output_test/right2.mp4"
                if not video_path:
                    print(f"Defult path: {video_path}")
                
                if video_path: 
                    config.source = video_path
                    break
                else: 
                    print("ERROR")
            print(f"Using video file: {config.source}")
            break
        else:
            print("\nchoice 1 or 2.\n")

    print(f"\nStarting detection")
    main(config)
