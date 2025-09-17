import os
import sys
import glob
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np
from ultralytics import YOLO

# -------------------- Classes to keep (COCO) --------------------
KEEP_IDS = [0, 1, 2, 3, 5, 9, 15, 16]
# ---------------------------------------------------------------

@dataclass
class Config:
    model_path: str = "yolo11n.pt"
    source: str = "0"
    conf_thresh: float = 0.50
    imgsz: int = 320
    detect_interval: int = 1
    drop_before_retrieve: int = 2

    resolution: Optional[Tuple[int, int]] = (640, 480)
    show_fps: bool = True
    show_boxes: bool = True

    request_fps: int = 30
    force_mjpeg: bool = False

    torch_threads: int = 2
    opencv_threads: int = 1


class CameraThread:
    def __init__(self, cap: cv2.VideoCapture, drop: int = 2):
        self.cap = cap
        self.drop = max(0, int(drop))
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.th = threading.Thread(target=self._update, daemon=True)
        self.th.start()

    def _update(self):
        while not self.stopped:
            for _ in range(self.drop):
                self.cap.grab()
            ok, f = self.cap.retrieve()
            if not ok:
                ok, f = self.cap.read()
                if not ok:
                    time.sleep(0.001)
                    continue
            with self.lock:
                self.frame = f

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.stopped = True
        self.th.join(timeout=0.5)


def try_open_usb(name_or_idx: str, res: Optional[Tuple[int, int]], fps: int, force_mjpeg: bool) -> cv2.VideoCapture:
    if name_or_idx.isdigit():
        usb_idx = int(name_or_idx)
    else:
        digits = "".join([c for c in name_or_idx if c.isdigit()])
        usb_idx = int(digits) if digits else 0

    attempts = [
        (cv2.CAP_V4L2, 'MJPG') if force_mjpeg else (cv2.CAP_V4L2, None),
        (cv2.CAP_V4L2, 'YUYV'),
        (cv2.CAP_ANY,  'MJPG') if force_mjpeg else (cv2.CAP_ANY, None),
        (cv2.CAP_ANY,  'YUYV'),
        (cv2.CAP_ANY,  None),
    ]
    last_err = None

    for backend, fourcc in attempts:
        cap = cv2.VideoCapture(usb_idx, backend)
        if not cap.isOpened():
            cap.release()
            last_err = f"cannot open (backend={backend})"
            continue

        if res is not None:
            w, h = res
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(w))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
        if fps:
            cap.set(cv2.CAP_PROP_FPS, int(fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if fourcc == 'MJPG':
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        elif fourcc == 'YUYV':
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

        ok, frame = cap.read()
        if ok and frame is not None:
            return cap

        last_err = f"opened but no frame (backend={backend}, fourcc={fourcc})"
        cap.release()

    raise RuntimeError(f"USB cam open failed: {last_err}")


def detect_source_type(src: str) -> str:
    img_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    vid_ext = {'.avi', '.mov', '.mp4', '.mkv', '.wmv', '.MP4', '.MOV'}

    if os.path.isdir(src):
        return 'folder'
    if os.path.isfile(src):
        _, ext = os.path.splitext(src)
        if ext in img_ext:
            return 'image'
        if ext in vid_ext:
            return 'video'
        raise ValueError(f"Unsupported file extension: {ext}")

    if src.isdigit() or src.startswith("/dev/video"):
        return 'usb'

    raise ValueError(f"Invalid source: {src}")


def warmup_model(model: YOLO, imgsz: int):
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    _ = model.predict(dummy, imgsz=imgsz, classes=KEEP_IDS, verbose=False)


def put_text(img: np.ndarray, text: str, y: int):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def draw_detections(img: np.ndarray, boxes: np.ndarray, confs: np.ndarray, clss: np.ndarray, labels, keep_ids: List[int]):
    bbox_colors = [
        (164,120, 87), ( 68,148,228), ( 93, 97,209), (178,182,133),
        ( 88,159,106), ( 96,202,231), (159,124,168), (169,162,241),
        ( 98,118,150), (172,176,184)
    ]
    for (xmin, ymin, xmax, ymax), c, k in zip(boxes, confs, clss):
        color = bbox_colors[int(k) % len(bbox_colors)]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        name = labels[k] if isinstance(labels, dict) else str(labels[k])
        text = f'{name} {int(c*100)}%'
        cv2.putText(img, text, (xmin, max(15, ymin-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(img, f'Objects (8 classes only): {len(boxes)}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def main(cfg: Config):
    os.environ.setdefault('OMP_NUM_THREADS', str(cfg.torch_threads))
    os.environ.setdefault('OPENBLAS_NUM_THREADS', str(cfg.torch_threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(cfg.torch_threads))
    try:
        import torch
        torch.set_num_threads(cfg.torch_threads)
    except Exception:
        pass
    try:
        cv2.setNumThreads(int(cfg.opencv_threads))
    except Exception:
        pass

    if not os.path.exists(cfg.model_path):
        print(f'ERROR: model not found -> {cfg.model_path}')
        sys.exit(1)

    model = YOLO(cfg.model_path, task='detect')
    labels = model.names
    warmup_model(model, cfg.imgsz)

    src_type = detect_source_type(cfg.source)

    cam_thread = None
    cap = None
    img_list = []
    resW, resH = (cfg.resolution if cfg.resolution else (None, None))
    do_resize = cfg.resolution is not None

    if src_type == 'image':
        img_list = [cfg.source]
    elif src_type == 'folder':
        for p in glob.glob(os.path.join(cfg.source, '*')):
            ext = os.path.splitext(p)[1].lower()
            if ext in {'.jpg', '.jpeg', '.png', '.bmp'}:
                img_list.append(p)
        img_list.sort()
    elif src_type == 'video':
        cap = cv2.VideoCapture(cfg.source)
        if do_resize:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  resW)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    elif src_type == 'usb':
        cap = try_open_usb(cfg.source, cfg.resolution, cfg.request_fps, cfg.force_mjpeg)
        cam_thread = CameraThread(cap, drop=cfg.drop_before_retrieve)

    # caches
    last_boxes = np.empty((0, 4), dtype=int)
    last_confs = np.empty((0,), dtype=float)
    last_clss  = np.empty((0,), dtype=int)
    frame_idx = 0

    # FPS collectors
    loop_fps_hist: List[float] = []
    det_fps_hist: List[float]  = []
    fps_avg_len = 200

    # ---- RealFPS (wall-clock) ----
    real_frames = 0
    real_start  = time.time()
    realfps_update_period = 1.0  # sec
    real_fps = 0.0

    # print tracking every 1s
    last_print_ts = time.time()
    print_period = 1.0

    while True:
        loop_t0 = time.perf_counter()

        # read frame
        if src_type in ('image', 'folder'):
            if frame_idx >= len(img_list):
                print('All images processed. Exit.')
                break
            path = img_list[frame_idx]
            frame = cv2.imread(path)
            if frame is None:
                frame_idx += 1
                continue
        elif src_type == 'video':
            ok, frame = cap.read()
            if not ok or frame is None:
                print('Video ended. Exit.')
                break
        else:  # usb
            frame = cam_thread.read()
            if frame is None:
                time.sleep(0.005)
                continue

        if do_resize:
            frame = cv2.resize(frame, (resW, resH))

        # detect at interval
        run_detect = (frame_idx % max(1, cfg.detect_interval) == 0)
        if run_detect:
            det_t0 = time.perf_counter()
            results = model.predict(frame, imgsz=cfg.imgsz, classes=KEEP_IDS, verbose=False)
            det_t1 = time.perf_counter()

            det_dt = det_t1 - det_t0
            det_fps = 1.0 / max(1e-6, det_dt)
            det_fps_hist.append(det_fps)
            if len(det_fps_hist) > fps_avg_len:
                det_fps_hist.pop(0)

            det = results[0].boxes
            if len(det) > 0:
                boxes = det.xyxy.cpu().numpy().astype(int)
                confs = det.conf.cpu().numpy()
                clss  = det.cls.cpu().numpy().astype(int)
                mask = (confs >= cfg.conf_thresh) & np.isin(clss, KEEP_IDS)
                last_boxes = boxes[mask]
                last_confs = confs[mask]
                last_clss  = clss[mask]
            else:
                last_boxes = np.empty((0, 4), dtype=int)
                last_confs = np.empty((0,), dtype=float)
                last_clss  = np.empty((0,), dtype=int)

        # draw
        if cfg.show_boxes and len(last_boxes) > 0:
            draw_detections(frame, last_boxes, last_confs, last_clss, labels, KEEP_IDS)

        # ---- Loop FPS (whole pipeline loop) ----
        loop_t1 = time.perf_counter()
        loop_dt = loop_t1 - loop_t0
        loop_fps = 1.0 / max(1e-6, loop_dt)
        loop_fps_hist.append(loop_fps)
        if len(loop_fps_hist) > fps_avg_len:
            loop_fps_hist.pop(0)

        # ---- RealFPS update (wall-clock) ----
        real_frames += 1
        elapsed = time.time() - real_start
        if elapsed >= realfps_update_period:
            real_fps = real_frames / max(1e-6, elapsed)
            real_frames = 0
            real_start = time.time()

        # overlays
        if cfg.show_fps:
            det_fps_avg  = float(np.mean(det_fps_hist)) if det_fps_hist else 0.0
            loop_fps_avg = float(np.mean(loop_fps_hist)) if loop_fps_hist else 0.0
            put_text(frame, f'DetFPS (detect only): {det_fps_avg:0.2f}', y=22)
            put_text(frame, f'LoopFPS (pipeline): {loop_fps_avg:0.2f}', y=46)
            put_text(frame, f'RealFPS (wall-clock): {real_fps:0.2f}', y=70)  # << NEW

        cv2.imshow("YOLO realtime (8 classes, clean skeleton)", frame)

        # print tracking every 1 second (uses last_boxes to be robust)
        now_ts = time.time()
        if now_ts - last_print_ts >= print_period:
            print(f"tracking : {len(last_boxes)}")
            last_print_ts = now_ts

        key = cv2.waitKey(1 if src_type in ('video', 'usb') else 0)
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('p'), ord('P')) and src_type in ('video', 'usb'):
            while True:
                k2 = cv2.waitKey(0)
                if k2 in (ord('p'), ord('P'), ord('q'), ord('Q'), 27):
                    if k2 in (ord('q'), ord('Q'), 27):
                        key = k2
                    break

        frame_idx += 1

        if src_type in ('image', 'folder'):
            if key in (ord('q'), ord('Q'), 27):
                break

    try:
        if cap is not None and src_type in ('video', 'usb'):
            if cam_thread:
                cam_thread.release()
            cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cfg = Config(
        model_path="yolo11n.pt",
        source="0",
        conf_thresh=0.50,
        imgsz=320,
        detect_interval=1,
        drop_before_retrieve=2,
        resolution=(640, 480),
        show_fps=True,
        show_boxes=True,
        request_fps=60,
        force_mjpeg=False,
        torch_threads=2,
        opencv_threads=1,
    )
    main(cfg)
