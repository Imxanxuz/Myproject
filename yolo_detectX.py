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

# --- สถานะการขับขี่ (State Machine) ---
class DriveState:
    NORMAL    = "NORMAL"    # วิ่งตามเลนปกติ
    STOPPING  = "STOPPING"  # พบวัตถุและหยุดรอ
    EVADING   = "EVADING"   # หักพวงมาลัยหลบออก
    BYPASSING = "BYPASSING" # วิ่งแซงขนานไปกับวัตถุ
    RETURNING = "RETURNING" # หักกลับเข้าเลนเดิม

@dataclass
class Config:
    model_path: str = "/home/rpi/yolo/yolo11n.pt"
    source: any = "/home/rpi/yolo/video_output_test/right2.mp4"
    resolution: tuple = (640, 480)
    conf_thresh: float = 0.25
    imgsz: int = 192
    
    # พารามิเตอร์การควบคุม
    STEERING_RANGE: tuple = (60.0, 120.0)
    STEERING_CENTER: float = 90.0
    PID_GAINS: tuple = (0.007, 0.012, 0.004)
    
    # ระยะตัดสินใจ (เมตร)
    DIST_CRITICAL: float = 1.8   # ระยะเริ่มหักหลบ
    DIST_SAFE_PASS: float = 3.5  # ระยะที่พ้นวัตถุแล้ว
    
    # เวลาในแต่ละสถานะ (วินาที)
    WAIT_TIME: float = 0.5       # หยุดนิ่งก่อนหลบ
    EVADE_DURATION: float = 1.2  # เวลาในการหักหัวรถ
    
    TARGET_CLASSES: list = field(default_factory=lambda: ["person", "car", "bus", "bicycle", "motorcycle"])
    focal_length: float = 500.0
    OBJECT_REAL_HEIGHTS: dict = field(default_factory=lambda: {
        "person": 1.75, "car": 1.5, "bus": 3.2, "bicycle": 1.0, "motorcycle": 1.2
    })

class MotorControl:
    def __init__(self, pins=(23, 24, 5, 6, 27, 27), freq=100):
        # ขา 23,24 ขับเคลื่อน | ขา 5,6 เลี้ยว | ขา 27 PWM
        self.IN1, self.IN2, self.IN3, self.IN4, self.ENA, self.ENB = pins
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(list(pins), GPIO.OUT)
        
        self.pwm_drive = GPIO.PWM(self.ENA, freq)
        self.pwm_drive.start(0)
        self.current_angle = 90.0
        self.current_speed = 0

    def drive(self, angle, speed=60):
        self.current_angle = angle
        self.current_speed = speed
        diff = angle - 90.0
        
        # ขับเคลื่อนล้อหลัง
        GPIO.output(self.IN1, GPIO.HIGH); GPIO.output(self.IN2, GPIO.LOW)
        
        # บังคับเลี้ยวล้อหน้า
        if diff < -5: # เลี้ยวซ้าย
            GPIO.output(self.IN3, GPIO.HIGH); GPIO.output(self.IN4, GPIO.LOW)
        elif diff > 5: # เลี้ยวขวา
            GPIO.output(self.IN3, GPIO.LOW); GPIO.output(self.IN4, GPIO.HIGH)
        else: # ล้อตรง
            GPIO.output(self.IN3, GPIO.LOW); GPIO.output(self.IN4, GPIO.LOW)
            
        self.pwm_drive.ChangeDutyCycle(speed)

    def stop(self):
        self.pwm_drive.ChangeDutyCycle(0)
        GPIO.output([self.IN1, self.IN2, self.IN3, self.IN4], GPIO.LOW)

    def cleanup(self):
        self.stop()
        GPIO.cleanup()

# --- ส่วนของการตรวจจับเลนและ PID (สรุปย่อเพื่อความกระชับ) ---
class LaneAssistant:
    def __init__(self, cfg):
        self.cfg = cfg
        # ... (เพิ่ม Logic Lane Detection และ PID ของคุณที่นี่) ...

def main(cfg: Config):
    motor = MotorControl()
    cap = cv2.VideoCapture(cfg.source)
    
    # การสื่อสารกับ YOLO Thread
    frame_q = Queue(maxsize=1); results_q = Queue(maxsize=1); stop_event = threading.Event()
    yolo_thread = threading.Thread(target=yolo_worker, args=(cfg, frame_q, results_q, stop_event))
    yolo_thread.start()

    # ข้อมูลระบบ
    current_state = DriveState.NORMAL
    motor_enabled = False
    lane_enabled = False
    last_boxes = []
    state_start_time = 0
    evade_direction = 0 # -1:ซ้าย, 1:ขวา

    def draw_panel(frame, title, lines, origin):
        x, y = origin
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + 220, y + (len(lines)+1)*22), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, title, (x+10, y+20), 0, 0.6, (0,255,255), 2)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x+10, y+45 + i*20), 0, 0.5, (255,255,255), 1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            view = cv2.resize(frame, cfg.resolution)
            h, w = view.shape[:2]

            # ส่งเฟรมไปให้ YOLO
            if frame_q.empty(): frame_q.put(view.copy())
            
            # รับผลลัพธ์จาก YOLO
            try:
                last_boxes = results_q.get_nowait()
            except Empty: pass

            # วิเคราะห์สิ่งกีดขวางในเลน
            lane_objs = [b for b in last_boxes if b['status'] == 'In Lane']
            min_dist = min([b['dist'] for b in lane_objs]) if lane_objs else 999
            
            # --- ระบบตัดสินใจอัตโนมัติ (Autonomous Logic) ---
            if motor_enabled:
                if current_state == DriveState.NORMAL:
                    if min_dist < cfg.DIST_CRITICAL:
                        current_state = DriveState.STOPPING
                        state_start_time = time.time()
                    else:
                        motor.drive(90) # ตัวอย่างวิ่งตรง (ใช้ PID จริงในส่วนนี้)

                elif current_state == DriveState.STOPPING:
                    motor.stop()
                    if time.time() - state_start_time > cfg.WAIT_TIME:
                        # เช็กเลนว่าง
                        left_free = not any(b['center'][0] < w*0.3 for b in last_boxes)
                        right_free = not any(b['center'][0] > w*0.7 for b in last_boxes)
                        
                        if right_free: evade_direction = 1; current_state = DriveState.EVADING
                        elif left_free: evade_direction = -1; current_state = DriveState.EVADING
                        state_start_time = time.time()

                elif current_state == DriveState.EVADING:
                    # หักหัวหลบ
                    motor.drive(90 + (evade_direction * 30), speed=50)
                    if time.time() - state_start_time > cfg.EVADE_DURATION:
                        current_state = DriveState.BYPASSING

                elif current_state == DriveState.BYPASSING:
                    motor.drive(90, speed=50) # วิ่งแซง
                    if min_dist > cfg.DIST_SAFE_PASS: # พ้นแล้ว
                        current_state = DriveState.RETURNING
                        state_start_time = time.time()

                elif current_state == DriveState.RETURNING:
                    # หักกลับเข้าเลน
                    motor.drive(90 - (evade_direction * 30), speed=50)
                    if time.time() - state_start_time > cfg.EVADE_DURATION:
                        current_state = DriveState.NORMAL

            else:
                motor.stop()

            # --- แสดงผล Dashboard (Panel) ---
            info = [
                f"STATE: {current_state}",
                f"MIN DIST: {min_dist:.1f}m",
                f"SPEED: {motor.current_speed}%",
                f"STEER: {motor.current_angle:.1f}"
            ]
            draw_panel(view, "DRIVING DASHBOARD", info, (10, 10))

            cv2.imshow("Autonomous System", view)
            key = cv2.waitKey(1)
            if key == ord('m'): motor_enabled = not motor_enabled
            if key == ord('q'): break

    finally:
        stop_event.set()
        motor.cleanup()
        cap.release()

def yolo_worker(cfg, f_q, r_q, s_ev):
    model = YOLO(cfg.model_path)
    while not s_ev.is_set():
        try:
            img = f_q.get(timeout=1)
            res = model.predict(img, imgsz=cfg.imgsz, verbose=False)
            boxes = []
            for b in res[0].boxes:
                # Logic คำนวณระยะทางและสถานะ (In Lane / Out Lane)
                pass 
            r_q.put(boxes)
        except: continue

if __name__ == "__main__":
    main(Config())
