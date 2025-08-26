from typing import List, Tuple

import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore


class YoloDetector:
    def __init__(self, weights_path: str, confidence_threshold: float = 0.25, device: str = "cpu", tracker: str = "bytetrack.yaml") -> None:
        if YOLO is None:
            raise RuntimeError("Ultralytics not installed. Run: pip install ultralytics")
        try:
            self.model = YOLO(weights_path)
            self.tracker = tracker
            self.confidence_threshold = confidence_threshold
            self.device = device
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO weights '{weights_path}': {e}")

    def detect_and_track(self, frame: np.ndarray) -> List[Tuple[int, int, float, Tuple[int, int, int, int]]]:
        results = self.model.track(frame, conf=self.confidence_threshold, device=self.device, tracker=self.tracker, persist=True)
        tracks: List[Tuple[int, int, float, Tuple[int, int, int, int]]] = []
        for r in results:
            if r.boxes.id is not None:
                boxes = r.boxes
                for i, b in enumerate(boxes):
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                    conf = float(b.conf[0].item())
                    cls = int(b.cls[0].item())
                    track_id = int(b.id[0].item()) if b.id is not None else i
                    tracks.append((track_id, cls, conf, (x1, y1, x2, y2)))
        return tracks

    def detect(self, frame: np.ndarray) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
        results = self.model.predict(frame, conf=self.confidence_threshold, device=self.device)
        detections: List[Tuple[int, float, Tuple[int, int, int, int]]] = []
        for r in results:
            boxes = r.boxes
            for b in boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                conf = float(b.conf[0].item())
                cls = int(b.cls[0].item())
                detections.append((cls, conf, (x1, y1, x2, y2)))
        return detections


