import os
import sys
import argparse
import glob
import time
import threading
import cv2
import numpy as np
from ultralytics import YOLO

# -------------------- Keep only 8 classes --------------------
# COCO IDs: person(0), bicycle(1), car(2), motorcycle(3), bus(5),
# traffic light(9), cat(15), dog(16)
KEEP_IDS = [0, 1, 2, 3, 5, 9, 15, 16]
# -------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='YOLO model path (.pt)')
    p.add_argument('--source', required=True, help='file/folder/video or "usb0" alias')
    p.add_argument('--thresh', type=float, default=0.5, help='confidence threshold')
    p.add_argument('--resolution', default=None, help='WxH display/output (e.g. 640x480)')
    p.add_argument('--record', action='store_true', help='record to demo1.avi (needs --resolution)')
    p.add_argument('--imgsz', type=int, default=320, help='inference size (e.g. 256/288/320/384)')
    p.add_argument('--torch-threads', type=int, default=2, help='torch.set_num_threads() & BLAS env')
    p.add_argument('--opencv-threads', type=int, default=1, help='cv2.setNumThreads()')
    # camera/video tuning
    p.add_argument('--mjpeg', action='store_true', help='force MJPEG on USB cam (good for USB2/RPi)')
    p.add_argument('--fps', type=int, default=30, help='request camera FPS')
    p.add_argument('--drop', type=int, default=2, help='grab-and-drop old frames before retrieve')
    # speed tricks
    p.add_argument('--detect-interval', type=int, default=1,
                   help='run detection every N frames (>=1); skipped frames reuse last result')
    p.add_argument('--nodraw', action='store_true', help='skip drawing boxes (measure pure inference)')
    return p.parse_args()

# ---------- Camera thread for low-latency grabbing ----------
class CameraThread:
    def __init__(self, cap, drop=2):
        self.cap = cap
        self.drop = max(0, drop)
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.th = threading.Thread(target=self.update, daemon=True)
        self.th.start()

    def update(self):
        while not self.stopped:
            # drop queued frames to keep only the freshest one
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
# ------------------------------------------------------------

def main():
    args = parse_args()

    # tame BLAS/OMP & Torch threads (helps stability on small CPUs like RPi)
    os.environ.setdefault('OMP_NUM_THREADS', str(args.torch_threads))
    os.environ.setdefault('OPENBLAS_NUM_THREADS', str(args.torch_threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(args.torch_threads))
    try:
        import torch
        torch.set_num_threads(args.torch_threads)
    except Exception:
        pass

    # OpenCV threads (too many can hurt on small SoCs)
    try:
        cv2.setNumThreads(int(args.opencv_threads))
    except Exception:
        pass

    model_path = args.model
    img_source  = args.source
    min_thresh  = float(args.thresh)
    user_res    = args.resolution
    record      = args.record
    imgsz       = int(args.imgsz)
    detect_every= max(1, int(args.detect-interval)) if hasattr(args, 'detect-interval') else max(1, int(args.detect_interval))

    if not os.path.exists(model_path):
        print('ERROR: invalid model path.')
        sys.exit(0)

    cv2.setUseOptimized(True)

    # Load model
    model = YOLO(model_path, task='detect')
    labels = model.names  # dict {id:name}

    # (important) warmup once at target imgsz to avoid first-frame stall
    _ = model.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), imgsz=imgsz,
                      classes=KEEP_IDS, verbose=False)

    # Detect source type
    img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
    vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

    if os.path.isdir(img_source):
        source_type = 'folder'
    elif os.path.isfile(img_source):
        _, ext = os.path.splitext(img_source)
        if ext in img_ext_list:
            source_type = 'image'
        elif ext in vid_ext_list:
            source_type = 'video'
        else:
            print(f'Unsupported file extension: {ext}')
            sys.exit(0)
    elif 'usb' in img_source:
        source_type = 'usb'
        usb_idx = int(img_source[3:])  # usb0 -> 0
    elif 'picamera' in img_source:
        source_type = 'picamera'
        picam_idx = int(img_source[8:])
    else:
        print(f'Invalid source: {img_source}')
        sys.exit(0)

    # Display resolution
    resize = False
    if user_res:
        resize = True
        resW, resH = map(int, user_res.split('x'))

    # Recording setup
    if record:
        if source_type not in ['video','usb']:
            print('Recording only for video/usb sources.')
            sys.exit(0)
        if not user_res:
            print('Please set --resolution to record.')
            sys.exit(0)
        record_name = 'demo1.avi'
        record_fps = args.fps
        recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

    # Load/init source
    cam_thread = None
    if source_type == 'image':
        imgs_list = [img_source]

    elif source_type == 'folder':
        imgs_list = []
        for file in glob.glob(os.path.join(img_source, '*')):
            _, file_ext = os.path.splitext(file)
            if file_ext in img_ext_list:
                imgs_list.append(file)

    elif source_type in ('video', 'usb'):
        cap_arg = img_source if source_type == 'video' else usb_idx
        backend = cv2.CAP_V4L2 if source_type == 'usb' else cv2.CAP_ANY
        cap = cv2.VideoCapture(cap_arg, backend)

        if user_res:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  resW)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
        if args.fps:
            cap.set(cv2.CAP_PROP_FPS, args.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if args.mjpeg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        cam_thread = CameraThread(cap, drop=args.drop)

    elif source_type == 'picamera':
        from picamera2 import Picamera2
        cap = Picamera2()
        if not user_res:
            resW, resH = 640, 480
            resize = True
        cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
        cap.start()

    # Colors
    bbox_colors = [
        (164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
        (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)
    ]

    avg_frame_rate = 0.0
    frame_rate_buffer = []
    fps_avg_len = 200
    img_count = 0

    # cache last detections for interval skipping
    last_boxes = np.empty((0, 4), dtype=int)
    last_confs = np.empty((0,), dtype=float)
    last_clss  = np.empty((0,), dtype=int)

    frame_idx = 0

    while True:
        t_start = time.perf_counter()

        # Grab frame
        if source_type in ('image','folder'):
            if img_count >= len(imgs_list):
                print('All images processed. Exiting.')
                break
            img_filename = imgs_list[img_count]
            frame = cv2.imread(img_filename)
            img_count += 1

        elif source_type == 'video':
            frame = cam_thread.read() if cam_thread else None
            if frame is None:
                ret, frame = cap.read()
                if not ret:
                    print('Reached end of video. Exiting.')
                    break

        elif source_type == 'usb':
            frame = cam_thread.read()
            if frame is None:
                print('Unable to read from camera. Exiting.')
                break

        elif source_type == 'picamera':
            frame_rgb = cap.capture_array()
            if frame_rgb is None:
                print('Unable to read from Picamera. Exiting.')
                break
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if resize:
            frame = cv2.resize(frame, (resW, resH))

        run_detect = (frame_idx % detect_every == 0)
        if run_detect:
            # -------- Inference (limit 8 classes + reduced imgsz) --------
            results = model.predict(frame, imgsz=imgsz, classes=KEEP_IDS, verbose=False)
            det = results[0].boxes

            if len(det) > 0:
                boxes = det.xyxy.cpu().numpy().astype(int)
                confs = det.conf.cpu().numpy()
                clss  = det.cls.cpu().numpy().astype(int)

                mask = (confs >= min_thresh) & np.isin(clss, KEEP_IDS)
                last_boxes = boxes[mask]
                last_confs = confs[mask]
                last_clss  = clss[mask]
            else:
                last_boxes = np.empty((0,4), dtype=int)
                last_confs = np.empty((0,), dtype=float)
                last_clss  = np.empty((0,), dtype=int)

        object_count = int(len(last_boxes))

        # Draw (optional)
        if not args.nodraw and object_count > 0:
            for (xmin, ymin, xmax, ymax), c, k in zip(last_boxes, last_confs, last_clss):
                color = bbox_colors[k % len(bbox_colors)]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                name = labels[k] if isinstance(labels, dict) else str(labels[k])
                text = f'{name} {int(c*100)}%'
                cv2.putText(frame, text, (xmin, max(15, ymin-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # FPS overlay
        if source_type in ('video','usb','picamera') and not args.nodraw:
            cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}',
                        (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

        # Display
        if not args.nodraw:
            cv2.putText(frame, f'Objects (8 classes): {object_count}',
                        (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
            cv2.imshow('YOLO (fast path, 8 classes)', frame)

        if record and not args.nodraw:
            recorder.write(frame)

        key = cv2.waitKey(1 if source_type in ('video','usb','picamera') else 0)
        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('s'), ord('S')):
            cv2.waitKey()
        elif key in (ord('p'), ord('P')) and not args.nodraw:
            cv2.imwrite('capture.png', frame)

        # FPS calc
        t_stop = time.perf_counter()
        frame_rate_calc = float(1.0 / max(1e-6, (t_stop - t_start)))
        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
        avg_frame_rate = float(np.mean(frame_rate_buffer))

        frame_idx += 1

    # Clean up
    print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
    if source_type in ('video','usb'):
        if cam_thread:
            cam_thread.release()
        try:
            cap.release()
        except Exception:
            pass
    elif source_type == 'picamera':
        cap.stop()
    if record:
        recorder.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
