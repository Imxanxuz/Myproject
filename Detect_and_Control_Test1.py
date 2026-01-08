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

# Config
GPIO.setwarnings(False)
GPIO_MODE = True
@dataclass
class Config:
    model_path: str = "/home/rpi/yolo/yolo11n.pt"
    source: any = 0
    resolution: tuple = (840,480)
    conf_thresh: float = 0.3
    imgsz: int = 224
    
    # --- [NEW] Steering & PID Control ---
    STEERING_RANGE: tuple = (60.0, 120.0)  # (Min=Left, Max=Right)
    STEERING_CENTER: float = 90.0
    # (Kp, Ki, Kd) 
    PID_GAINS: tuple = (0.006, 0.015, 0.004) # (P, I, D)
    PID_WINDUP_LIMIT: float = 100.0  

    # --- [MODIFIED] Lane Detection Tuning ---
    roi_top_ratio: float = 0.67
    roi_top_left_x_ratio: float = 0.43
    roi_top_right_x_ratio: float = 0.55
    canny_low: int = 50
    canny_high: int = 150
    min_lane_slope: float = 0.25
    poly_fit_deque_len: int = 4  # ลด deque ลง เพราะ PID จะช่วยเรื่องความนิ่งแทน
    poly_fit_margin: int = 75
    poly_min_points_for_fit: int = 10
    
    # Lane Sanity Check Parameters ---
    LANE_SANITY_CHECK_PX: tuple = (250, 700)
    LANE_SANITY_CHECK_RATIO: tuple = (0.7, 3.0)

    # ROI Bottom ---
    roi_bottom_left_x_ratio: float = 0.1  # 10% from left edge
    roi_bottom_right_x_ratio: float = 0.85  # 70% from left edge (10% from right)

    #Other Parameters
    focal_length: float = 1700.0
    normal_speed: int = 30
    # detection_distance_m
    detection_distance_m: float = 75.0 
    lane_origin_y_ratio: float = 0.75 # Y-level for PID calculation
    
    TARGET_CLASSES: list = field(default_factory=lambda: ["person", "car", "bus", "bicycle", "motorcycle"])
    
    LANE_DIST_CALIB_M: tuple = (10, 80)
    lane_mask_margin: int = 20
    OBJECT_REAL_HEIGHTS: dict = field(default_factory=lambda: {
        "person": 1.7, "car": 1.5, "truck": 3.5, "bus": 3.2, "bicycle": 1.0, "motorcycle": 1.2
    })
    DEFAULT_OBJECT_HEIGHT: float = 1.5

#class MotorControl:
    def __init__(self, pin_b, pin_c, freq=50, steering_range=(60.0, 120.0)):
        self.pin_b = pin_b
        self.pin_c = pin_c
        self.freq = freq
        self.steering_range = steering_range
        self.center_angle = (steering_range[0] + steering_range[1]) / 2.0 # Should be 90.0
        
        self.min_duty = 6.667
        self.center_duty = 7.5
        self.max_duty = 8.333
        
        self.servo_pwm1 = None
        self.servo_pwm2 = None
        self.current_angle = self.center_angle
        
        if GPIO_MODE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin_b, GPIO.OUT)
            GPIO.setup(self.pin_c, GPIO.OUT)
            self.servo_pwm1 = GPIO.PWM(self.pin_b, self.freq)
            self.servo_pwm2 = GPIO.PWM(self.pin_c, self.freq)
            self.servo_pwm1.start(self.center_duty)
            self.servo_pwm2.start(self.center_duty)

    def _angle_to_duty(self, angle: float) -> float:
        angle = np.clip(angle, self.steering_range[0], self.steering_range[1])
        return np.interp(
            angle,
            [self.steering_range[0], self.center_angle, self.steering_range[1]],
            [self.min_duty, self.center_duty, self.max_duty]
        )

    def move_to(self, angle: float):
        self.current_angle = angle
        duty = self._angle_to_duty(self.current_angle)
        if GPIO_MODE:
            self.servo_pwm1.ChangeDutyCycle(duty)
            self.servo_pwm2.ChangeDutyCycle(duty)

    def set_stop(self):
        if GPIO_MODE:
            self.servo_pwm1.ChangeDutyCycle(0)
            self.servo_pwm2.ChangeDutyCycle(0)
        self.current_angle = self.center_angle 

    def stop(self):
        if GPIO_MODE:
            self.servo_pwm1.stop()
            self.servo_pwm2.stop()
            GPIO.cleanup()
class MotorControl_to_MotorDriver:
    def __init__(self,IN1 = 23, IN2 = 24,ENA = None,ENB = None,IN3 = 25, IN4 = 26, freq=100, steering_range=(60.0, 120.0)):
        self.IN1 = IN1
        self.IN2 = IN2
        self.IN3 = IN3
        self.IN4 = IN4
        self.ENA = ENA
        self.ENB = ENB
        self.freq = freq
        self.steering_range = steering_range
        self.center_angle = (steering_range[0] + steering_range[1]) / 2.0 # Should be 90.0
        
        self.min_duty = 6.667
        self.center_duty = 7.5
        self.max_duty = 8.333
        
        self.servo_pwm1 = None
        self.servo_pwm2 = None
        self.current_angle = self.center_angle
        
        if GPIO_MODE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.IN1, GPIO.OUT)
            GPIO.setup(self.IN2, GPIO.OUT)
            GPIO.setup(self.IN3, GPIO.OUT)
            GPIO.setup(self.IN4, GPIO.OUT)
            GPIO.setup(self.ENA, GPIO.OUT)
            GPIO.setup(self.ENB, GPIO.OUT)
            self.servo_pwm1 = GPIO.PWM(self.ENA, self.freq)
            self.servo_pwm2 = GPIO.PWM(self.ENB, self.freq)
            self.servo_pwm1.start(self.center_duty)
            self.servo_pwm2.start(self.center_duty)

    def _angle_to_duty(self, angle: float) -> float:
        angle = np.clip(angle, self.steering_range[0], self.steering_range[1])
        return np.interp(
            angle,
            [self.steering_range[0], self.center_angle, self.steering_range[1]],
            [self.min_duty, self.center_duty, self.max_duty]
        )
        
    def move_to(self, angle: float):
        self.current_angle = angle
        duty = self._angle_to_duty(self.current_angle)
        if GPIO_MODE:
            self.servo_pwm1.ChangeDutyCycle(duty)
            self.servo_pwm2.ChangeDutyCycle(duty)
    
    def forward(self,speed):
        if GPIO_MODE:
            GPIO.output(self.IN1, GPIO.HIGH)
            GPIO.output(self.IN2, GPIO.LOW)
            GPIO.output(self.IN3, GPIO.HIGH)
            GPIO.output(self.IN4, GPIO.LOW)
            self.servo_pwm1.ChangeDutyCycle(speed)
            self.servo_pwm2.ChangeDutyCycle(speed)
        

    def set_stop(self):
        if GPIO_MODE:
            self.servo_pwm1.ChangeDutyCycle(0)
            self.servo_pwm2.ChangeDutyCycle(0)
        self.current_angle = self.center_angle 

    def stop(self):
        if GPIO_MODE:
            self.servo_pwm1.stop()
            self.servo_pwm2.stop()
            GPIO.cleanup()

    try :
        forward(50)
        sleep(1)
        stop()
        GPIO.cleanup()
    except KeyboardInterrupt:
        GPIO.cleanup()

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

# --- Main Application (MODIFIED panels, object logic) ---
def main(cfg: Config):
    lane_detector = LaneDetector(cfg)
    keeper = LaneKeeper(cfg)
    #motor = MotorControl(pin_b=19, pin_c=13, steering_range=cfg.STEERING_RANGE)
    motor = MotorControl_to_MotorDriver(ENA = 9,ENB = 19,steering_range = cfg.STEERING_RANGE)
    
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
    lane_fps, yolo_fps, total_fps = 0, 0, 0
    
    last_motor_update_time = 0.0

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

            t_lane_start = time.time()
            left_fit, right_fit = lane_detector.detect_lanes(display_frame)
            lane_time = time.time() - t_lane_start
            if lane_time > 0: lane_fps = 1.0 / lane_time

            lane_mask = np.zeros(display_frame.shape[:2], dtype=np.uint8)
            if left_fit is not None and right_fit is not None:
                lane_mask = lane_detector.get_lane_area_mask(display_frame.shape, left_fit, right_fit)

            steering_angle = cfg.STEERING_CENTER
            
            if lane_detection_enabled:
                steering_angle, _, _ = keeper.calculate_steering(
                    (left_fit, right_fit), display_frame.shape
                )
                display_frame = lane_detector.draw_lanes(display_frame, left_fit, right_fit)
                
                if left_fit is not None and right_fit is not None:
                    overlay = np.zeros_like(display_frame)
                    plot_y = np.linspace(int(h * cfg.roi_top_ratio), h - 1, 20)
                    left_fit_x = lane_detector._get_x_at_y(left_fit, plot_y)
                    right_fit_x = lane_detector._get_x_at_y(right_fit, plot_y)
                    if left_fit_x is not None and right_fit_x is not None:
                        pts_left = np.asarray([left_fit_x, plot_y]).T
                        pts_right = np.asarray([right_fit_x, plot_y]).T
                        points = np.vstack([pts_left, np.flipud(pts_right)]).astype(np.int32)
                        cv2.fillPoly(overlay, [points], (255, 255, 255))
                        display_frame = cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0)
            
            try:
                new_results, yolo_time = results_queue.get_nowait()
                if yolo_time > 0: yolo_fps = 1.0 / yolo_time
                temp_boxes = []
                for box in new_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = new_results[0].names[int(box.cls[0])]
                    dist = estimate_distance(cfg, y2 - y1, label)
                    bcx, bcy = (x1 + x2) // 2, y2
                    
                    if 0 <= bcy < h and 0 <= bcx < w and lane_mask[bcy, bcx] == 255:
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
            
            loop_time = time.time() - t_total_start
            total_fps = 1.0 / loop_time if loop_time > 0 else 0
            
            perf_lines = [f"Total: {total_fps:.1f} FPS", 
                          f"Yaw  : {lane_fps:.1f} FPS", 
                          f"YOLO : {yolo_fps:.1f} FPS"]
            draw_panel(display_frame, "PERFORMANCE", perf_lines, (w - 170, 10), 160)
            
            if lane_detection_enabled and motor_enable:
                driving_lines = [f"Rec. Speed : {cfg.normal_speed} km/h", 
                                 f"Rec. Angle : {steering_angle:.1f}", 
                                 f"Motor Angle: {motor.current_angle:.1f}"]
                draw_panel(display_frame, "DRIVING INFO", driving_lines, (10, 10), 200)
            else:
                status_lines = [f"Lane Assist: {'ON' if lane_detection_enabled else 'OFF'}", f"Motor       : {'ON' if motor_enable else 'OFF'}"]
                draw_panel(display_frame, "SYSTEM STATUS", status_lines, (10, 10), 200)
            
            in_lane_labels = [data['label'] for data in last_known_boxes if data['status'] == 'In Lane']
            out_lane_labels = [data['label'] for data in last_known_boxes if data['status'] == 'Out of Lane']
            in_str = ", ".join([f"{c} {l}" for l, c in Counter(in_lane_labels).items()])
            out_str = ", ".join([f"{c} {l}" for l, c in Counter(out_lane_labels).items()])
            obj_lines = [f"In Lane : {in_str or 'None'}", 
                         f"Out Lane: {out_str or 'None'}"]
            draw_panel(display_frame, "DETECTED OBJECTS", obj_lines, (10, h - (len(obj_lines)*20 + 35)), 250)

            # --- [MODIFIED] Requests 4, 5, 6: Visualization moved to roi_top_ratio ---
            if lane_detection_enabled and left_fit is not None and right_fit is not None:
                
                # [MODIFIED] Set visualization Y-level to the top of the ROI
                y_viz = int(h * cfg.roi_top_ratio) 
                
                x_left_viz = keeper._x_at_y(left_fit, y_viz)
                x_right_viz = keeper._x_at_y(right_fit, y_viz)

                if x_left_viz is not None and x_right_viz is not None:
                    # Request 4: Horizontal line at top of ROI
                    cv2.line(display_frame, (int(x_left_viz), y_viz), (int(x_right_viz), y_viz), (0, 255, 255), 2)
                    
                    # Request 5: Deviation visualization at top of ROI
                    x_center_viz = (x_left_viz + x_right_viz) // 2
                    deviation_viz = x_center_viz - (w // 2)
                    
                    cv2.line(display_frame, (w // 2, y_viz), (int(x_center_viz), y_viz), (0, 0, 255), 3) 
                    cv2.circle(display_frame, (int(x_center_viz), y_viz), 8, (0, 255, 0), -1) 
                    cv2.putText(display_frame, f"Deviation: {int(deviation_viz)}px", (w // 2 - 50, y_viz - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Request 6: Arctan/Steering line (from bottom-center to top-viz-center)
                    cv2.line(display_frame, (w // 2, h - 1), (int(x_center_viz), y_viz), (255, 0, 255), 2)

                # Draw vertical center line (from user's previous code, now dynamic to y_viz)
                cv2.line(display_frame, (w // 2, y_viz), (w // 2, h - 1), (255, 100, 0), 2)
                cv2.putText(display_frame, "Image Center", (w // 2 + 5, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            # --- [END MODIFIED] ---
            
            if motor_enable:
                current_time = time.time()
                if (current_time - last_motor_update_time) >= 0.5:
                    motor.move_to(steering_angle)
                    last_motor_update_time = current_time
                # (No 'else' needed, motor holds its last position)
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
                video_path = "/home/rpi/yolo/video_output_test/right2.mp4"
                if not video_path:
                    print(f"[INFO] No path entered, using default: {video_path}")
                
                if video_path: 
                    config.source = video_path
                    break
                else: 
                    print("[ERROR] Path cannot be empty. Please try again.")
            print(f"[INFO] Using video file: {config.source}")
            break
        else:
            print("\n[ERROR] Invalid choice. Please enter 1 or 2.\n")

    print(f"\n[INFO] Starting detection...")
    main(config)
