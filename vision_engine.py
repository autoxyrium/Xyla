"""
XYLA Vision Engine
==================
Core computer vision module for the Xyla smart mirror.
Runs on Raspberry Pi 5 using OpenCV.

Handles:
- Face detection + tracking
- Facial landmark detection (eyes, lips, nose, jawline)
- Makeup region analysis
- Outfit region capture
- Frame streaming to the UI
- Triggers for AI analysis

Author: Xyla Project
"""

import cv2
import numpy as np
import base64
import json
import time
import threading
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format='[XYLA CV] %(levelname)s: %(message)s')
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════

@dataclass
class FaceData:
    """Data about a detected face in the frame."""
    x: int
    y: int
    w: int
    h: int
    confidence: float
    landmarks: Optional[dict] = None
    look_score: Optional[float] = None  # 0-100 overall readiness score

    @property
    def center(self):
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def area(self):
        return self.w * self.h

    def to_dict(self):
        return asdict(self)


@dataclass
class MirrorState:
    """Current state of what the mirror sees."""
    face_detected: bool = False
    face_data: Optional[FaceData] = None
    person_present: bool = False
    lighting_quality: str = "unknown"   # good / low / harsh
    frame_timestamp: float = 0.0
    capture_count: int = 0


# ═══════════════════════════════════════════════════════════
# VISION ENGINE
# ═══════════════════════════════════════════════════════════

class XylaVisionEngine:
    """
    Core vision engine for Xyla.
    
    Usage:
        engine = XylaVisionEngine()
        engine.start()
        
        # In your app loop:
        frame = engine.get_frame()
        state = engine.get_state()
        
        # Capture for AI analysis:
        image_b64 = engine.capture_for_ai()
        
        engine.stop()
    """

    # Haar cascade paths — works on RPi with opencv-python
    CASCADE_FACE    = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    CASCADE_EYE     = cv2.data.haarcascades + "haarcascade_eye.xml"
    CASCADE_SMILE   = cv2.data.haarcascades + "haarcascade_smile.xml"

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        mirror_flip: bool = True,        # Mirror the image (like a real mirror)
        save_dir: str = "captures",
        presence_threshold: int = 3,     # Seconds before declaring "person present"
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.mirror_flip = mirror_flip
        self.save_dir = save_dir
        self.presence_threshold = presence_threshold

        # State
        self.state = MirrorState()
        self._frame = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._capture_thread = None
        self._cap = None

        # Detectors
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_FACE)
        self.eye_cascade  = cv2.CascadeClassifier(self.CASCADE_EYE)
        self.smile_cascade = cv2.CascadeClassifier(self.CASCADE_SMILE)

        # Try to load dlib landmarks if available (better quality)
        self._dlib_available = False
        self._landmark_predictor = None
        self._dlib_detector = None
        self._try_load_dlib()

        # Presence tracking
        self._last_face_time = 0.0
        self._presence_announced = False

        # Callbacks
        self.on_person_detected = None    # fn(face_data)
        self.on_person_left = None        # fn()
        self.on_frame_ready = None        # fn(frame)

        # Ensure save dir exists
        os.makedirs(self.save_dir, exist_ok=True)

        log.info(f"Vision engine initialised — {width}x{height} @ {fps}fps")
        log.info(f"dlib landmarks: {'YES (high quality)' if self._dlib_available else 'NO (using Haar cascade)'}")

    def _try_load_dlib(self):
        """Attempt to load dlib for better landmark detection."""
        try:
            import dlib
            model_path = "models/shape_predictor_68_face_landmarks.dat"
            if os.path.exists(model_path):
                self._dlib_detector = dlib.get_frontal_face_detector()
                self._landmark_predictor = dlib.shape_predictor(model_path)
                self._dlib_available = True
                log.info("dlib loaded — 68-point facial landmarks active")
            else:
                log.info(f"dlib model not found at {model_path} — download it for better landmarks")
                log.info("Run: python src/vision/download_models.py")
        except ImportError:
            log.info("dlib not installed — pip install dlib for better landmark detection")

    # ──────────────────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────────────────

    def start(self):
        """Start the camera and processing thread."""
        log.info(f"Opening camera index {self.camera_index}...")
        self._cap = cv2.VideoCapture(self.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")

        # Warmup
        for _ in range(5):
            self._cap.read()

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(f"Camera opened — actual resolution: {actual_w}x{actual_h}")

        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        log.info("Vision engine started")

    def stop(self):
        """Stop the camera and processing thread."""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        log.info("Vision engine stopped")

    # ──────────────────────────────────────────────────────
    # CAPTURE LOOP (runs in background thread)
    # ──────────────────────────────────────────────────────

    def _capture_loop(self):
        """Main capture and processing loop."""
        frame_time = 1.0 / self.fps
        face_detect_interval = 5   # Detect every N frames (performance)
        frame_count = 0

        while self._running:
            loop_start = time.time()

            ret, frame = self._cap.read()
            if not ret:
                log.warning("Failed to read frame — retrying...")
                time.sleep(0.1)
                continue

            # Mirror flip
            if self.mirror_flip:
                frame = cv2.flip(frame, 1)

            frame_count += 1

            # Run face detection every N frames
            if frame_count % face_detect_interval == 0:
                self._process_face(frame)

            # Draw overlays onto frame
            annotated = self._draw_overlays(frame.copy())

            # Store frame (thread-safe)
            with self._frame_lock:
                self._frame = annotated

            # Callback
            if self.on_frame_ready:
                self.on_frame_ready(annotated)

            # Frame rate control
            elapsed = time.time() - loop_start
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ──────────────────────────────────────────────────────
    # FACE DETECTION + ANALYSIS
    # ──────────────────────────────────────────────────────

    def _process_face(self, frame: np.ndarray):
        """Detect faces and update mirror state."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Equalise histogram for better detection in varying lighting
        gray_eq = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) > 0:
            # Pick the largest face (closest to mirror)
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

            face_data = FaceData(
                x=int(x), y=int(y), w=int(w), h=int(h),
                confidence=self._estimate_confidence(faces),
            )

            # Landmark detection
            if self._dlib_available:
                face_data.landmarks = self._detect_landmarks_dlib(gray, x, y, w, h)
            else:
                face_data.landmarks = self._detect_landmarks_haar(gray, x, y, w, h)

            # Lighting quality
            self.state.lighting_quality = self._analyse_lighting(frame, x, y, w, h)

            self.state.face_detected = True
            self.state.face_data = face_data
            self.state.frame_timestamp = time.time()
            self._last_face_time = time.time()

            # Person presence detection
            if not self._presence_announced:
                self._presence_announced = True
                self.state.person_present = True
                if self.on_person_detected:
                    threading.Thread(
                        target=self.on_person_detected,
                        args=(face_data,),
                        daemon=True
                    ).start()
                log.info("Person detected at mirror")
        else:
            self.state.face_detected = False

            # Check if person has left
            if self._presence_announced:
                time_since_face = time.time() - self._last_face_time
                if time_since_face > self.presence_threshold:
                    self._presence_announced = False
                    self.state.person_present = False
                    self.state.face_data = None
                    if self.on_person_left:
                        threading.Thread(target=self.on_person_left, daemon=True).start()
                    log.info("Person left mirror")

    def _detect_landmarks_haar(self, gray: np.ndarray, x, y, w, h) -> dict:
        """Basic landmark detection using Haar cascades (fallback)."""
        roi = gray[y:y+h, x:x+w]
        landmarks = {}

        # Eyes
        eyes = self.eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        if len(eyes) >= 2:
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            landmarks['left_eye'] = {
                'x': int(x + eyes_sorted[0][0] + eyes_sorted[0][2]//2),
                'y': int(y + eyes_sorted[0][1] + eyes_sorted[0][3]//2)
            }
            landmarks['right_eye'] = {
                'x': int(x + eyes_sorted[1][0] + eyes_sorted[1][2]//2),
                'y': int(y + eyes_sorted[1][1] + eyes_sorted[1][3]//2)
            }
        elif len(eyes) == 1:
            landmarks['eye_1'] = {
                'x': int(x + eyes[0][0] + eyes[0][2]//2),
                'y': int(y + eyes[0][1] + eyes[0][3]//2)
            }

        # Smile detection
        smiles = self.smile_cascade.detectMultiScale(
            roi[h//2:], scaleFactor=1.7, minNeighbors=20, minSize=(25, 15)
        )
        landmarks['smile_detected'] = len(smiles) > 0

        # Estimate nose and mouth from face geometry
        landmarks['nose_approx'] = {
            'x': int(x + w//2),
            'y': int(y + h * 0.6)
        }
        landmarks['mouth_approx'] = {
            'x': int(x + w//2),
            'y': int(y + h * 0.75)
        }

        return landmarks

    def _detect_landmarks_dlib(self, gray: np.ndarray, x, y, w, h) -> dict:
        """68-point landmark detection using dlib (high quality)."""
        import dlib

        rect = dlib.rectangle(x, y, x + w, y + h)
        shape = self._landmark_predictor(gray, rect)
        pts = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        # Map to named landmarks
        return {
            'jaw': pts[0:17],
            'left_eyebrow': pts[17:22],
            'right_eyebrow': pts[22:27],
            'nose_bridge': pts[27:31],
            'nose_tip': pts[31:36],
            'left_eye': pts[36:42],
            'right_eye': pts[42:48],
            'outer_lips': pts[48:60],
            'inner_lips': pts[60:68],
            # Derived points
            'left_eye_center': self._centroid(pts[36:42]),
            'right_eye_center': self._centroid(pts[42:48]),
            'mouth_center': self._centroid(pts[48:68]),
            'nose_center': pts[30],
            'smile_detected': self._estimate_smile(pts),
            'eye_symmetry': self._eye_symmetry_score(pts),
        }

    def _centroid(self, points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (int(sum(xs)/len(xs)), int(sum(ys)/len(ys)))

    def _estimate_smile(self, pts) -> bool:
        """Estimate smile from lip corner positions."""
        left_corner  = pts[48]
        right_corner = pts[54]
        top_lip      = pts[51]
        # If corners are higher than centre top lip, likely smiling
        avg_corner_y = (left_corner[1] + right_corner[1]) / 2
        return bool(avg_corner_y < top_lip[1] + 5)

    def _eye_symmetry_score(self, pts) -> float:
        """Score eye symmetry 0-1 (1 = perfectly symmetric)."""
        left_eye_h  = abs(pts[37][1] - pts[41][1])
        right_eye_h = abs(pts[43][1] - pts[47][1])
        if max(left_eye_h, right_eye_h) == 0:
            return 1.0
        return float(min(left_eye_h, right_eye_h) / max(left_eye_h, right_eye_h))

    def _estimate_confidence(self, faces) -> float:
        """Estimate detection confidence from number of detections."""
        return min(1.0, len(faces) * 0.5 + 0.5)

    def _analyse_lighting(self, frame: np.ndarray, x, y, w, h) -> str:
        """Analyse lighting quality in the face region."""
        face_roi = frame[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        mean_brightness = np.mean(gray_roi)
        std_brightness  = np.std(gray_roi)

        if mean_brightness < 60:
            return "low"
        elif mean_brightness > 200:
            return "harsh"
        elif std_brightness > 60:
            return "uneven"
        else:
            return "good"

    # ──────────────────────────────────────────────────────
    # OVERLAY DRAWING
    # ──────────────────────────────────────────────────────

    def _draw_overlays(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI overlays onto the frame."""
        h, w = frame.shape[:2]

        # Corner brackets (Xyla UI style)
        bracket_len = 30
        bracket_thickness = 2
        cyan = (255, 220, 0)    # BGR — gold/cyan look

        corners = [
            ((20, 20), (20+bracket_len, 20), (20, 20+bracket_len)),           # TL
            ((w-20, 20), (w-20-bracket_len, 20), (w-20, 20+bracket_len)),     # TR
            ((20, h-20), (20+bracket_len, h-20), (20, h-20-bracket_len)),     # BL
            ((w-20, h-20), (w-20-bracket_len, h-20), (w-20, h-20-bracket_len)), # BR
        ]

        for corner_pts in corners:
            origin, h_pt, v_pt = corner_pts
            cv2.line(frame, origin, h_pt, cyan, bracket_thickness, cv2.LINE_AA)
            cv2.line(frame, origin, v_pt, cyan, bracket_thickness, cv2.LINE_AA)

        # Face detection overlay
        if self.state.face_detected and self.state.face_data:
            fd = self.state.face_data
            self._draw_face_overlay(frame, fd)

        # Lighting warning
        if self.state.lighting_quality == "low":
            self._draw_text_overlay(frame, "LOW LIGHT — MOVE CLOSER TO LAMP", (w//2, h-40), color=(0, 165, 255))
        elif self.state.lighting_quality == "harsh":
            self._draw_text_overlay(frame, "HARSH LIGHT DETECTED", (w//2, h-40), color=(0, 165, 255))

        # Timestamp (small, corner)
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, ts, (w-100, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

        return frame

    def _draw_face_overlay(self, frame: np.ndarray, fd: FaceData):
        """Draw face detection box and landmarks."""
        # Face bounding box — subtle green
        cv2.rectangle(frame,
                       (fd.x, fd.y),
                       (fd.x + fd.w, fd.y + fd.h),
                       (0, 255, 100), 1, cv2.LINE_AA)

        # "FACE DETECTED" label
        label = "FACE LOCKED"
        cv2.putText(frame, label,
                    (fd.x, fd.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 100), 1, cv2.LINE_AA)

        # Landmarks
        if fd.landmarks:
            self._draw_landmarks(frame, fd.landmarks)

    def _draw_landmarks(self, frame: np.ndarray, landmarks: dict):
        """Draw facial landmarks."""
        dot_color = (100, 255, 200)
        dot_size = 2

        if 'left_eye_center' in landmarks:
            cv2.circle(frame, landmarks['left_eye_center'], dot_size, dot_color, -1, cv2.LINE_AA)
        if 'right_eye_center' in landmarks:
            cv2.circle(frame, landmarks['right_eye_center'], dot_size, dot_color, -1, cv2.LINE_AA)
        if 'nose_center' in landmarks:
            cv2.circle(frame, landmarks['nose_center'], dot_size, dot_color, -1, cv2.LINE_AA)
        if 'mouth_center' in landmarks:
            cv2.circle(frame, landmarks['mouth_center'], dot_size, dot_color, -1, cv2.LINE_AA)

        # Draw lip outline if available
        if 'outer_lips' in landmarks:
            pts = np.array(landmarks['outer_lips'], np.int32)
            cv2.polylines(frame, [pts], True, (150, 100, 255), 1, cv2.LINE_AA)

        # Draw eye outlines
        for eye_key in ['left_eye', 'right_eye']:
            if eye_key in landmarks and isinstance(landmarks[eye_key], list):
                pts = np.array(landmarks[eye_key], np.int32)
                cv2.polylines(frame, [pts], True, dot_color, 1, cv2.LINE_AA)

    def _draw_text_overlay(self, frame, text, pos, color=(0, 255, 200)):
        x, y = pos
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x - tw//2 - 6, y - th - 6), (x + tw//2 + 6, y + 6),
                      (0, 0, 0), -1)
        cv2.putText(frame, text, (x - tw//2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # ──────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest annotated frame (thread-safe)."""
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    def get_state(self) -> MirrorState:
        """Get current mirror state."""
        return self.state

    def capture_for_ai(self, region: str = "full") -> Optional[str]:
        """
        Capture current frame as base64 JPEG for Claude Vision API.

        Args:
            region: "full" | "face" | "outfit" | "eyes" | "lips"

        Returns:
            Base64-encoded JPEG string, or None if no frame available.
        """
        frame = self.get_frame()
        if frame is None:
            return None

        if region != "full" and self.state.face_data:
            fd = self.state.face_data
            frame = self._crop_region(frame, fd, region)

        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def _crop_region(self, frame: np.ndarray, fd: FaceData, region: str) -> np.ndarray:
        """Crop frame to a specific region of interest."""
        h, w = frame.shape[:2]

        if region == "face":
            # Add 30% padding around face
            pad_x = int(fd.w * 0.3)
            pad_y = int(fd.h * 0.3)
            x1 = max(0, fd.x - pad_x)
            y1 = max(0, fd.y - pad_y)
            x2 = min(w, fd.x + fd.w + pad_x)
            y2 = min(h, fd.y + fd.h + pad_y)
            return frame[y1:y2, x1:x2]

        elif region == "outfit":
            # Bottom 60% of frame (body)
            return frame[int(h * 0.35):, :]

        elif region == "eyes" and fd.landmarks:
            # Crop around eye region
            lm = fd.landmarks
            eye_y = fd.y + int(fd.h * 0.25)
            return frame[eye_y:eye_y + int(fd.h * 0.25), fd.x:fd.x + fd.w]

        elif region == "lips" and fd.landmarks:
            # Crop around mouth region
            mouth_y = fd.y + int(fd.h * 0.65)
            return frame[mouth_y:mouth_y + int(fd.h * 0.25), fd.x:fd.x + fd.w]

        return frame

    def save_capture(self, label: str = "capture") -> str:
        """Save current frame to disk. Returns filepath."""
        frame = self.get_frame()
        if frame is None:
            raise RuntimeError("No frame available to save")

        self.state.capture_count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"xyla_{label}_{ts}_{self.state.capture_count:04d}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        log.info(f"Saved capture: {filepath}")
        return filepath

    def get_face_summary(self) -> dict:
        """
        Get a text summary of current face state for AI context.
        Useful as additional context in Claude prompts.
        """
        if not self.state.face_detected or not self.state.face_data:
            return {"face_detected": False}

        fd = self.state.face_data
        summary = {
            "face_detected": True,
            "lighting_quality": self.state.lighting_quality,
            "face_size_pct": round((fd.area / (self.width * self.height)) * 100, 1),
        }

        if fd.landmarks:
            lm = fd.landmarks
            summary["smile_detected"] = lm.get("smile_detected", False)
            if "eye_symmetry" in lm:
                summary["eye_symmetry_score"] = round(lm["eye_symmetry"], 2)

        return summary

    def run_preview(self):
        """
        Run a local preview window. Useful for testing on desktop.
        Press 'q' to quit, 's' to save a capture.
        """
        self.start()
        log.info("Preview window open — press 'q' to quit, 's' to save")

        try:
            while True:
                frame = self.get_frame()
                if frame is not None:
                    cv2.imshow("XYLA Mirror Preview", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    path = self.save_capture("preview")
                    print(f"Saved: {path}")

        finally:
            self.stop()
            cv2.destroyAllWindows()