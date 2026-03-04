"""
Live YOLO Object Detection with SORT Tracking

Usage:
    python yolo_tracker.py -m best.pt -s 0                              # webcam
    python yolo_tracker.py -m best.pt -s "https://youtu.be/..."         # YouTube
    python yolo_tracker.py -m best.onnx -s video.mp4                    # local file
    python yolo_tracker.py -m best_openvino_model -s video.mp4          # OpenVINO folder

Supported formats: .pt, .onnx, OpenVINO folder (via ultralytics)
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

DEFAULT_CLASSES = ["person", "vehicle", "animal", "traffic_light", "traffic_sign"]

COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
    (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
]


# ---------------------------------------------------------------------------
# SORT Tracking
# ---------------------------------------------------------------------------

class KalmanBoxTracker:
    """Tracks a single object using a Kalman filter on bounding box state."""
    _id_counter = 0

    def __init__(self, bbox, class_id, confidence):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # Constant-velocity state transition
        self.kf.F = np.eye(7)
        self.kf.F[0, 4] = 1.0
        self.kf.F[1, 5] = 1.0
        self.kf.F[2, 6] = 1.0

        # Measurement: observe [cx, cy, area, ratio]
        self.kf.H = np.zeros((4, 7))
        np.fill_diagonal(self.kf.H, 1.0)

        # Covariance tuning (SORT paper defaults)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self._bbox_to_z(bbox).reshape(4, 1)

        KalmanBoxTracker._id_counter += 1
        self.id = KalmanBoxTracker._id_counter
        self.class_id = class_id
        self.confidence = confidence
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0

    @staticmethod
    def _bbox_to_z(bbox):
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        return np.array([bbox[0] + w / 2, bbox[1] + h / 2, w * h, w / max(h, 1e-6)])

    @staticmethod
    def _z_to_bbox(z):
        w = np.sqrt(max(z[2] * z[3], 0))
        h = z[2] / max(w, 1e-6) if w > 0 else 0
        return np.array([z[0] - w / 2, z[1] - h / 2, z[0] + w / 2, z[1] + h / 2])

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox, class_id, confidence):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.class_id = class_id
        self.confidence = confidence
        self.kf.update(self._bbox_to_z(bbox).reshape(4, 1))

    def get_state(self):
        return self._z_to_bbox(self.kf.x.flatten()[:4])


class SORTTracker:
    """Simple Online and Realtime Tracking (SORT)."""

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    @staticmethod
    def _iou_batch(bb_det, bb_trk):
        det = np.expand_dims(bb_det, 1)
        trk = np.expand_dims(bb_trk, 0)
        xx1 = np.maximum(det[..., 0], trk[..., 0])
        yy1 = np.maximum(det[..., 1], trk[..., 1])
        xx2 = np.minimum(det[..., 2], trk[..., 2])
        yy2 = np.minimum(det[..., 3], trk[..., 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_det = (det[..., 2] - det[..., 0]) * (det[..., 3] - det[..., 1])
        area_trk = (trk[..., 2] - trk[..., 0]) * (trk[..., 3] - trk[..., 1])
        return inter / np.maximum(area_det + area_trk - inter, 1e-6)

    def update(self, detections):
        # Predict new locations
        predicted = []
        to_remove = []
        for i, trk in enumerate(self.trackers):
            pred = trk.predict()
            if np.any(np.isnan(pred)):
                to_remove.append(i)
            else:
                predicted.append(pred)
        for i in reversed(to_remove):
            self.trackers.pop(i)

        # Associate detections to trackers via Hungarian algorithm
        unmatched_det = list(range(len(detections)))
        if detections and self.trackers:
            det_boxes = np.array([d[0] for d in detections])
            trk_boxes = np.array(predicted)
            iou_matrix = self._iou_batch(det_boxes, trk_boxes)

            if iou_matrix.size > 0:
                row_idx, col_idx = linear_sum_assignment(-iou_matrix)
                matched_d = set()
                for r, c in zip(row_idx, col_idx):
                    if iou_matrix[r, c] >= self.iou_threshold:
                        matched_d.add(r)
                        bbox, cls_id, conf = detections[r]
                        self.trackers[c].update(bbox, cls_id, conf)
                unmatched_det = [i for i in range(len(detections)) if i not in matched_d]

        # Create new trackers for unmatched detections
        for i in unmatched_det:
            bbox, cls_id, conf = detections[i]
            self.trackers.append(KalmanBoxTracker(bbox, cls_id, conf))

        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        # Return active tracks
        return [
            (trk.get_state(), trk.id, trk.class_id, trk.confidence)
            for trk in self.trackers
            if trk.hit_streak >= self.min_hits or trk.age <= self.min_hits
        ]


# ---------------------------------------------------------------------------
# Video Stream
# ---------------------------------------------------------------------------

class VideoStream:
    """Threaded capture for live streams, sequential for local files."""

    def __init__(self, source):
        backend = cv2.CAP_DSHOW if isinstance(source, int) and sys.platform == 'win32' else cv2.CAP_FFMPEG
        self.cap = cv2.VideoCapture(source, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        self.is_live = isinstance(source, int) or (
            isinstance(source, str) and source.startswith(('http://', 'https://', 'rtsp://'))
        )

        self.ret, self.frame = False, None
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = None

        for _ in range(10):
            self.ret, self.frame = self.cap.read()
            if self.ret and self.frame is not None:
                h, w = self.frame.shape[:2]
                total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = self.cap.get(cv2.CAP_PROP_FPS) or 0
                info = f"{w}x{h}"
                if total > 0:
                    info += f", {total} frames, {fps:.1f} fps"
                print(f"Stream ready ({info})")
                break
            time.sleep(0.5)
        if not self.ret:
            raise RuntimeError("Cannot read frames from source.")

        if self.is_live:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            time.sleep(0.5)  # let thread buffer a frame before main loop starts

    def _update(self):
        retries = 0
        max_retries = 30  # ~3 seconds of retries for stream hiccups
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                retries = 0
                with self.lock:
                    self.ret, self.frame = ret, frame
            else:
                retries += 1
                if retries > max_retries:
                    with self.lock:
                        self.ret = False
                    self.stopped = True
                    break
                time.sleep(0.1)

    def read(self):
        if self.is_live:
            with self.lock:
                return self.ret, self.frame.copy() if self.frame is not None else None
        self.ret, self.frame = self.cap.read()
        return self.ret, self.frame

    def release(self):
        self.stopped = True
        if self.thread:
            self.thread.join(timeout=2.0)
        self.cap.release()


# ---------------------------------------------------------------------------
# YOLO Detector
# ---------------------------------------------------------------------------

class YOLODetector:
    """YOLO detector via ultralytics — supports .pt, .onnx, and OpenVINO."""

    def __init__(self, model_path, conf=0.25, iou=0.45, imgsz=640, class_names=None):
        from ultralytics import YOLO

        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.class_names = class_names or DEFAULT_CLASSES

        path = os.path.abspath(model_path)

        # OpenVINO: ultralytics requires folder name ending with _openvino_model
        if os.path.isdir(path) and not path.endswith('_openvino_model'):
            new_path = path.rstrip(os.sep) + '_openvino_model'
            if not os.path.exists(new_path):
                os.rename(path, new_path)
                print(f"[openvino] Renamed folder to: {os.path.basename(new_path)}")
            path = new_path

        self.model = YOLO(path, task='detect')
        if hasattr(self.model, 'names') and self.model.names:
            self.class_names = list(self.model.names.values())

    def detect(self, frame):
        results = self.model(
            frame, imgsz=self.imgsz, conf=self.conf,
            iou=self.iou, verbose=False, device='cpu'
        )
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
                conf = float(r.boxes.conf[i].cpu())
                cls_id = int(r.boxes.cls[i].cpu())
                detections.append(([x1, y1, x2, y2], cls_id, conf))
        return detections


# ---------------------------------------------------------------------------
# Source Resolution
# ---------------------------------------------------------------------------

def resolve_source(source):
    if source.isdigit():
        return int(source)
    if any(x in source for x in ['youtube.com', 'youtu.be', 'twitch.tv']):
        return _extract_stream_url(source)
    return source


def _extract_stream_url(url):
    try:
        import streamlink
        streams = streamlink.streams(url)
        if streams:
            for quality in ['720p', '480p', 'best']:
                if quality in streams:
                    print(f"[streamlink] Using quality: {quality}")
                    return streams[quality].url
    except Exception:
        pass

    try:
        cmd = [
            'yt-dlp', '-f', 'best[height<=720][vcodec!*=av01]/best[height<=720]/best',
            '--no-warnings', '-g', url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("[yt-dlp] Stream URL resolved")
            return result.stdout.strip().split('\n')[0]
        raise RuntimeError(f"yt-dlp error: {result.stderr.strip()}")
    except FileNotFoundError:
        print("ERROR: yt-dlp not found. Install with: pip install yt-dlp", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_results(frame, tracks, class_names, fps):
    h, w = frame.shape[:2]

    for bbox, track_id, class_id, conf in tracks:
        x1 = max(0, min(int(bbox[0]), w - 1))
        y1 = max(0, min(int(bbox[1]), h - 1))
        x2 = max(0, min(int(bbox[2]), w - 1))
        y2 = max(0, min(int(bbox[3]), h - 1))
        color = COLORS[track_id % len(COLORS)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cls_name = class_names[class_id] if class_id < len(class_names) else f"cls_{class_id}"
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(y1 - th - 8, 0)
        cv2.rectangle(frame, (x1, ly), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (10, 10), (160, 45), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live YOLO Detection with SORT Tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', '-m', required=True,
                        help='Path to model (.pt, .onnx, or OpenVINO folder)')
    parser.add_argument('--source', '-s', default='0',
                        help='Video source: webcam id, file, YouTube URL, stream URL')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference resolution')
    parser.add_argument('--max-age', type=int, default=30, help='SORT: max lost frames')
    parser.add_argument('--min-hits', type=int, default=3, help='SORT: min hits to show')
    parser.add_argument('--sort-iou', type=float, default=0.3, help='SORT: match threshold')
    parser.add_argument('--skip', type=int, default=0, help='Skip N frames between inferences')
    parser.add_argument('--delay', type=int, default=1, help='Display delay ms (slow down)')
    parser.add_argument('--display-width', type=int, default=1280, help='Window width')
    parser.add_argument('--classes', nargs='+', default=None, help='Override class names')
    parser.add_argument('--output', '-o', default=None, help='Save to video file (.mp4)')
    args = parser.parse_args()

    # Resolve source
    print(f"Resolving source: {args.source}")
    source = resolve_source(args.source)
    print("Opening video stream...")

    # Init
    stream = VideoStream(source)
    detector = YOLODetector(args.model, args.conf, args.iou, args.imgsz, args.classes)
    tracker = SORTTracker(args.max_age, args.min_hits, args.sort_iou)
    class_names = args.classes or detector.class_names

    print(f"Model: {args.model}")
    print(f"Classes: {class_names}")

    # Display mode
    has_display = os.environ.get('DISPLAY') or sys.platform == 'win32'
    use_window = has_display and args.output is None

    if use_window:
        print("Press 'q' to quit, 'p' to pause/resume")
        cv2.namedWindow("YOLO Tracker", cv2.WINDOW_NORMAL)
    elif args.output:
        print(f"Saving to: {args.output}")
    else:
        print("No display. Use --output output.mp4 or Ctrl+C to stop.")

    # Frame timing
    video_fps = stream.cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_duration = 1.0 / video_fps
    next_frame_time = time.perf_counter()

    fps, fps_buf = 0.0, deque(maxlen=30)
    frame_count = 0
    tracks = []
    writer = None

    try:
        while True:
            t0 = time.perf_counter()

            ret, frame = stream.read()
            if not ret or frame is None:
                print("End of stream.")
                break
            frame_count += 1

            # Skip frames to maintain real-time pacing for local files
            if not stream.is_live and use_window and args.delay <= 1:
                while time.perf_counter() > next_frame_time + frame_duration:
                    ret2, frame2 = stream.read()
                    if not ret2 or frame2 is None:
                        break
                    frame = frame2
                    frame_count += 1
                    next_frame_time += frame_duration

            # Detection + tracking
            if args.skip == 0 or frame_count % (args.skip + 1) == 0:
                detections = detector.detect(frame)
                tracks = tracker.update(detections)
            else:
                tracks = [
                    (trk.get_state(), trk.id, trk.class_id, trk.confidence)
                    for trk in tracker.trackers
                    if trk.hit_streak >= tracker.min_hits
                ]

            # FPS
            elapsed = time.perf_counter() - t0
            fps_buf.append(1.0 / max(elapsed, 1e-6))
            fps = sum(fps_buf) / len(fps_buf)

            # Draw
            display = draw_results(frame, tracks, class_names, fps)
            dh, dw = display.shape[:2]
            if dw != args.display_width:
                display = cv2.resize(display, None, fx=args.display_width / dw,
                                     fy=args.display_width / dw)

            # Save to file
            if args.output:
                if writer is None:
                    h_out, w_out = display.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(args.output, fourcc, video_fps, (w_out, h_out))
                    print(f"Writing {w_out}x{h_out} @ {video_fps:.1f} fps")
                writer.write(display)

            # Display
            if use_window:
                cv2.imshow("YOLO Tracker", display)

                if not stream.is_live and args.delay <= 1:
                    next_frame_time += frame_duration
                    wait_ms = max(1, int((next_frame_time - time.perf_counter()) * 1000))
                else:
                    wait_ms = args.delay

                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    while cv2.waitKey(100) & 0xFF != ord('p'):
                        pass
                    next_frame_time = time.perf_counter()
            elif frame_count % 100 == 0:
                print(f"  Frame {frame_count} | FPS: {fps:.1f} | Tracks: {len(tracks)}")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    if writer:
        writer.release()
        print(f"Saved: {args.output}")
    stream.release()
    if use_window:
        cv2.destroyAllWindows()
    print(f"Done. {frame_count} frames processed.")


if __name__ == '__main__':
    main()
