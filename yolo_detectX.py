
#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import cv2
import numpy as np
import time
from dataclasses import dataclass
from ultralytics import YOLO

# CONFIGURATION
GPIO_MODE = False     

if GPIO_MODE:
    import motors as mot
    print("[GPIO MODE]  Motor control ENABLED")
else:
    print("[SIMULATION MODE]  Motor control DISABLED")


@dataclass
class Config:
    model_path: str = "C:/SUN/Y4.1/Project/code/yolo11n.pt"
    source: str = "0"                          
    resolution: tuple = (640,640)
    conf_thresh: float = 0.4
    imgsz: int = 120
    steering_gain: float = 0.02
    max_angle: float = 30.0
    roi_ratio: float = 0.6
    canny_low: int = 75
    canny_high: int = 150
    focal_length: float = 700.0
    normal_speed: int = 50


# LANE DETECTOR
class LaneDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.prev_lanes = []

    def detect_lanes(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.cfg.canny_low, self.cfg.canny_high)

        h, w = edges.shape
        mask = np.zeros_like(edges)
        pts = np.array([[(
            int(w * 0.1), h),
            (int(w * 0.4), int(h * self.cfg.roi_ratio)),
            (int(w * 0.6), int(h * self.cfg.roi_ratio)),
            (int(w * 0.9), h)
        ]], np.int32)
        cv2.fillPoly(mask, pts, 255)
        roi = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 30,
                                minLineLength=40, maxLineGap=30)
        if lines is None:
            return self.prev_lanes

        left, right = [], []
        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.4 or abs(slope) > 1.0:
                continue
            if slope < 0:
                left.append((x1, y1, x2, y2))
            else:
                right.append((x1, y1, x2, y2))

        lanes = []
        if left:
            avg_left = self._average_lines(left, h)
            if avg_left is not None:
                lanes.append(avg_left)
        if right:
            avg_right = self._average_lines(right, h)
            if avg_right is not None:
                lanes.append(avg_right)

        if lanes:
            self.prev_lanes = lanes
        return lanes

    def _average_lines(self, lines, height):
        xs, ys = [], []
        for x1, y1, x2, y2 in lines:
            xs += [x1, x2]
            ys += [y1, y2]
        if len(xs) < 2:
            return None
        slope, intercept = np.polyfit(xs, ys, 1)
        y1, y2 = height, int(height * self.cfg.roi_ratio)
        x1, x2 = int((y1 - intercept) / slope), int((y2 - intercept) / slope)
        return [x1, y1, x2, y2]

    def draw_lanes(self, frame, lanes):
        out = frame.copy()
        for x1, y1, x2, y2 in lanes:
            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 255), 6)
        return out

    def get_lane_area_mask(self, shape, lanes):
        mask = np.zeros(shape[:2], dtype=np.uint8)
        if len(lanes) == 2:
            pts = np.array([[(
                lanes[0][0], lanes[0][1]),
                (lanes[0][2], lanes[0][3]),
                (lanes[1][2], lanes[1][3]),
                (lanes[1][0], lanes[1][1])
            ]], np.int32)
            cv2.fillPoly(mask, pts, 255)
        return mask


# LANE KEEPER
class LaneKeeper:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.angle = 0.0

    @staticmethod
    def _x_at_y(line, y_ref):
        x1, y1, x2, y2 = line
        if x2 == x1:
            return x1
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return int((y_ref - intercept) / slope)

    def calculate_steering(self, lanes, frame_shape):
        h, w = frame_shape[:2]
        if len(lanes) < 2:
            return self.angle * self.cfg.max_angle, None, None

        lanes_sorted = sorted(lanes, key=lambda x: x[0])
        left, right = lanes_sorted[0], lanes_sorted[1]
        y_ref = int(h * 0.9)
        x_left = self._x_at_y(left, y_ref)
        x_right = self._x_at_y(right, y_ref)
        lane_center = int((x_left + x_right) / 2)
        image_center = int(w / 2)

        deviation = lane_center - image_center
        offset_px = deviation
        self.angle = np.clip(-deviation * self.cfg.steering_gain, -1, 1)
        return self.angle * self.cfg.max_angle, offset_px, (lane_center, y_ref)


# DISTANCE + SPEED FUNCTIONS
def estimate_distance(cfg: Config, box_h: int, label: str) -> float:
    real_height_map = {
        "person": 1.7, "car": 1.5, "truck": 3.0, "motorcycle": 1.2, "bicycle": 1.5
    }
    real_h = real_height_map.get(label, 1.5)
    if box_h <= 0:
        return np.inf
    return round((cfg.focal_length * real_h) / box_h, 2)


def suggest_speed(distance: float, cfg: Config) -> int:
    if distance < 3:
        return 0
    elif distance < 7:
        return 15
    elif distance < 15:
        return 30
    else:
        return cfg.normal_speed



# MAIN
def main(cfg: Config):
    global GPIO_MODE

    print("[INFO] Loading YOLO model...")
    model = YOLO(cfg.model_path)
    lane_detector = LaneDetector(cfg)
    keeper = LaneKeeper(cfg)

    cap = cv2.VideoCapture(int(cfg.source))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.resolution[1])

    frame_counter = 0
    print("[INFO] Press 'G' to toggle GPIO mode, 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_counter += 1
        start_time = time.time()

        lanes = lane_detector.detect_lanes(frame)
        frame = lane_detector.draw_lanes(frame, lanes)
        steering_angle, offset_px, lane_center_point = keeper.calculate_steering(lanes, frame.shape)

        h, w = frame.shape[:2]
        image_center = (w // 2, int(h * 0.9))
        cv2.line(frame, (image_center[0], 0), (image_center[0], h), (255, 255, 255), 1)
        cv2.circle(frame, image_center, 6, (255, 0, 0), -1)
        if lane_center_point is not None:
            cv2.circle(frame, lane_center_point, 6, (0, 255, 0), -1)
            cv2.line(frame, image_center, lane_center_point, (0, 255, 0), 2)

        results = model.predict(frame, imgsz=cfg.imgsz, conf=cfg.conf_thresh, verbose=False)
        det = results[0].boxes
        lane_mask = lane_detector.get_lane_area_mask(frame.shape, lanes)
        in_lane_objs, min_dist_in_lane = [], np.inf

        for box in det:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h_box = y2 - y1
            label = model.names[int(box.cls[0])]
            dist = estimate_distance(cfg, h_box, label)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            inside = lane_mask[cy, cx] == 255 if lanes else False
            color = (0, 255, 0) if inside else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {dist:.1f}m", (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if inside:
                in_lane_objs.append((label, dist))
                min_dist_in_lane = min(min_dist_in_lane, dist)

        suggested_speed = suggest_speed(min_dist_in_lane, cfg) if in_lane_objs else cfg.normal_speed
        status = f"Object IN lane ({min_dist_in_lane:.1f} m)" if in_lane_objs else "No object in lane"

        fps = 1.0 / (time.time() - start_time + 1e-6)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"{status}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Speed: {suggested_speed} km/h", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if suggested_speed > 0 else (0, 0, 255), 2)

        mode_text = f"GPIO MODE: {'ON' if GPIO_MODE else 'OFF'}"
        color = (0, 255, 0) if GPIO_MODE else (0, 0, 255)
        cv2.putText(frame, mode_text, (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


        if offset_px is not None:
            direction = "straight"
            if GPIO_MODE and frame_counter % 10 == 0:
                if -6 < offset_px < 6:
                    mot.frontmiddle(); mot.forward(25)
                    direction = "straight"
                elif offset_px > 6:
                    mot.frontleft(); mot.forward(25)
                    direction = "turn right"
                elif offset_px < -6:
                    mot.frontright(); mot.forward(25)
                    direction = "turn left"


            pwm_status = mot.get_pwm_status() if GPIO_MODE else {"A": 0, "B": 0, "C": 0, "D": 0}
            pwm_text = f"PWM A:{pwm_status['A']:.0f}%  B:{pwm_status['B']:.0f}%  C:{pwm_status['C']:.0f}%  D:{pwm_status['D']:.0f}%"
            cv2.putText(frame, pwm_text, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)
            cv2.putText(frame, f"Direction: {direction}", (10, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if frame_counter % 20 == 0 and GPIO_MODE:
                print(f"[MOTOR] {direction} | {pwm_text}")

            cv2.putText(frame, f"Steering: {steering_angle:.1f} degree ({direction})", (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Offset: {offset_px:+.1f}px", (10, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
        else:
            cv2.putText(frame, "Steering: N/A (lanes not found)", (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("YOLO + Lane + Speed + GPIO + PWM", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('g'):
            GPIO_MODE = not GPIO_MODE
            if GPIO_MODE:
                print("[GPIO MODE] ✅ Motor control ENABLED")
            else:
                print("[GPIO MODE] ❌ Motor control DISABLED")
                if 'mot' in globals():
                    mot.stop()

        elif key in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()
    if GPIO_MODE:
        mot.stop()
    print("[INFO] Exited successfully.")


if __name__ == "__main__":
    cfg = Config()
    main(cfg)
