#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
import threading
import RPi.GPIO as GPIO
from dataclasses import dataclass, field
from ultralytics import YOLO
from collections import deque, Counter
from queue import Queue, Empty

# --- ระบบสถานะการตัดสินใจ (Autonomous Drive States) ---
class DriveState:
    NORMAL    = "FOLLOW_LANE"  # วิ่งตามเลนปกติ
    STOPPING  = "SAFE_STOP"    # ตรวจพบวัตถุระยะใกล้และหยุดรอ
    EVADING   = "EVADE_OUT"    # หักพวงมาลัยเพื่อแซงออก
    BYPASSING = "BYPASS_OBJ"   # วิ่งขนานไปกับวัตถุที่แซง
    RETURNING = "RETURN_LANE"  # หักพวงมาลัยกลับเข้าเลนเดิม

@dataclass
class Config:
    # --- [MODIFIED] Source changed to 0 for Live Camera ---
    model_path: str = "/home/rpi/yolo/yolo11n.pt"
    source: any = 0 
    resolution: tuple = (640, 480)
    conf_thresh: float = 0.25
    imgsz: int = 192
    
    # การควบคุมมอเตอร์และ PID
    STEERING_RANGE: tuple = (60.0, 120.0)
    STEERING_CENTER: float = 90.0
    PID_GAINS: tuple = (0.007, 0.012, 0.004)
    PID_WINDUP_LIMIT: float = 100.0
    
    # เงื่อนไขระยะปลอดภัย (เมตร)
    DIST_CRITICAL: float = 1.8   # ระยะที่รถต้องเริ่มหยุด/หลบ
    DIST_SAFE_PASS: float = 3.8  # ระยะที่รถถือว่าแซงพ้นวัตถุแล้ว
    
    # เวลาในแต่ละ State (วินาที)
    WAIT_BEFORE_EVADE: float = 0.5
    EVADE_TIME: float = 1.2
    
    # การตั้งค่าเลน
    roi_top_ratio: float = 0.65
    canny_low: int = 50
    canny_high: int = 150
    lane_origin_y_ratio: float = 0.75
    
    TARGET_CLASSES: list = field(default_factory=lambda: ["person", "car", "bus", "bicycle", "motorcycle"])
    focal_length: float = 500.0
    OBJECT_REAL_HEIGHTS: dict = field(default_factory=lambda: {
        "person": 1.75, "car": 1.5, "bus": 3.2, "bicycle": 1.0, "motorcycle": 1.2
    })

class MotorControl:
    def __init__(self, pins=(23, 24, 5, 6, 27, 27), freq=100):
        # 23,24=Drive | 5,6=Steer | 27=PWM
        self.pins = pins
        self.IN1, self.IN2, self.IN3, self.IN4, self.ENA, self.ENB = pins
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(list(pins), GPIO.OUT)
        
        self.pwm = GPIO.PWM(self.ENA, freq)
        self.pwm.start(0)
        self.current_angle = 90.0
        self.current_speed = 0

    def drive(self, angle, speed=60):
        self.current_angle = angle
        self.current_speed = speed
        diff = angle - 90.0
        
        # ขับเคลื่อนล้อหลัง (Forward)
        GPIO.output(self.IN1, GPIO.HIGH); GPIO.output(self.IN2, GPIO.LOW)
        
        # บังคับเลี้ยวล้อหน้า (Digital Logic)
        if diff < -5:   # ซ้าย
            GPIO.output(self.IN3, GPIO.HIGH); GPIO.output(self.IN4, GPIO.LOW)
        elif diff > 5:  # ขวา
            GPIO.output(self.IN3, GPIO.LOW); GPIO.output(self.IN4, GPIO.HIGH)
        else:           # ตรง
            GPIO.output(self.IN3, GPIO.LOW); GPIO.output(self.IN4, GPIO.LOW)
            
        self.pwm.ChangeDutyCycle(speed)

    def set_stop(self):
        self.pwm.ChangeDutyCycle(0)
        GPIO.output([self.IN1, self.IN2, self.IN3, self.IN4], GPIO.LOW)

    def cleanup(self):
        self.set_stop()
        GPIO.cleanup()

# --- [Placeholder for LaneProcessor & PID Logic] ---
# (ส่วนนี้คงไว้ตามฟังก์ชันเดิมที่คุณมีสำหรับการประมวลผล Lane และ PID)

def main(cfg: Config):
    motor = MotorControl()
    cap = cv2.VideoCapture(cfg.source)
    
    # Threading สำหรับ YOLO
    frame_q = Queue(maxsize=1); res_q = Queue(maxsize=1); stop_ev = threading.Event()
    yolo_worker = threading.Thread(target=yolo_proc, args=(cfg, frame_q, res_q, stop_ev))
    yolo_worker.start()

    # ระบบจัดการ State
    current_state = DriveState.NORMAL
    motor_active = False
    lane_assist = False
    last_boxes = []
    state_start_t = 0
    evade_side = 1 # 1=ขวา, -1=ซ้าย

    def draw_panel_ui(img, title, lines, pos):
        x, y = pos
        h_box = (len(lines) + 1) * 22 + 10
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + 240, y + h_box), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        cv2.putText(img, title, (x+10, y+20), 0, 0.6, (0,255,255), 2)
        for i, text in enumerate(lines):
            cv2.putText(img, text, (x+10, y+45 + i*22), 0, 0.5, (255,255,255), 1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            view = cv2.resize(frame, cfg.resolution)
            h, w = view.shape[:2]

            # ส่งเฟรมให้ YOLO Thread
            if frame_q.empty(): frame_q.put(view.copy())
            
            # รับข้อมูลจาก YOLO
            try:
                last_boxes = res_q.get_nowait()
            except Empty: pass

            # เช็กวัตถุในเลน
            lane_objs = [b for b in last_boxes if b['status'] == 'In Lane']
            min_dist = min([b['dist'] for b in lane_objs]) if lane_objs else 99.0

            # --- Autonomous State Machine Logic ---
            if motor_active:
                if current_state == DriveState.NORMAL:
                    if min_dist < cfg.DIST_CRITICAL:
                        current_state = DriveState.STOPPING
                        state_start_t = time.time()
                    else:
                        # ขับตามเลนปกติ (เรียกใช้ PID steering_angle ที่นี่)
                        motor.drive(90, speed=65)

                elif current_state == DriveState.STOPPING:
                    motor.set_stop()
                    if time.time() - state_start_t > cfg.WAIT_BEFORE_EVADE:
                        # ตัดสินใจเลือกทางหลบ (สแกนพื้นที่ว่างข้างๆ)
                        left_free = not any(b['center'][0] < w*0.3 for b in last_boxes)
                        right_free = not any(b['center'][0] > w*0.7 for b in last_boxes)
                        
                        if right_free:
                            evade_side = 1; current_state = DriveState.EVADING
                        elif left_free:
                            evade_side = -1; current_state = DriveState.EVADING
                        state_start_t = time.time()

                elif current_state == DriveState.EVADING:
                    # หักหลบออก (Manual Angle)
                    motor.drive(90 + (evade_side * 35), speed=50)
                    if time.time() - state_start_t > cfg.EVADE_TIME:
                        current_state = DriveState.BYPASSING

                elif current_state == DriveState.BYPASSING:
                    motor.drive(90, speed=55) # วิ่งแซงขนาน
                    if min_dist > cfg.DIST_SAFE_PASS: # เมื่อพ้นระยะวัตถุ
                        current_state = DriveState.RETURNING
                        state_start_t = time.time()

                elif current_state == DriveState.RETURNING:
                    # หักกลับเข้าเลน
                    motor.drive(90 - (evade_side * 35), speed=50)
                    if time.time() - state_start_t > cfg.EVADE_TIME:
                        current_state = DriveState.NORMAL
            else:
                motor.set_stop()

            # --- Dashboard Visualization ---
            status_info = [
                f"DRIVE MODE: {current_state}",
                f"OBSTACLE: {min_dist:.1f} m",
                f"STEERING: {motor.current_angle:.1f}",
                f"THROTTLE: {motor.current_speed} %"
            ]
            draw_panel_ui(view, "AUTONOMOUS STATUS", status_info, (10, 10))

            cv2.imshow("Raspberry Pi 4 Autonomous Car", view)
            
            key = cv2.waitKey(1)
            if key == ord('q'): break
            if key == ord('m'): motor_active = not motor_active
            if key == ord('e'): lane_assist = not lane_assist

    finally:
        stop_ev.set()
        motor.cleanup()
        cap.release()
        cv2.destroyAllWindows()

def yolo_proc(cfg, f_q, r_q, s_ev):
    model = YOLO(cfg.model_path)
    while not s_ev.is_set():
        try:
            img = f_q.get(timeout=1)
            results = model.predict(img, imgsz=cfg.imgsz, verbose=False)
            detected = []
            # ... (Logic คำนวณระยะทางและระบุ In Lane/Out Lane) ...
            r_q.put(detected)
        except: continue

if __name__ == "__main__":
    main(Config())
