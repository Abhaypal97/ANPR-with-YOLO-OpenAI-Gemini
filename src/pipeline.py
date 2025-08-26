import base64
import time
from typing import Optional, List, Dict

import cv2

from .detector import YoloDetector
from .ocr_backends import DummyOCR, OpenAIOCR, GeminiOCR
from .storage import save_any
from .utils import load_config, now_iso


def _get_backend(name: str, cfg: dict):
    if name == "openai" and cfg.get("openai", {}).get("api_key"):
        return OpenAIOCR(cfg["openai"]["api_key"])  # type: ignore
    if name == "gemini" and cfg.get("gemini", {}).get("api_key"):
        return GeminiOCR(cfg["gemini"]["api_key"])  # type: ignore
    return DummyOCR()


def _bgr_crop_to_b64(image, box) -> str:
    x1, y1, x2, y2 = box
    crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    _, buf = cv2.imencode(".jpg", crop)
    return base64.b64encode(buf).decode("utf-8")


def _should_process_track(track_id: int, conf: float, ocr_cache: Dict[int, str], 
                         min_confidence: float = 0.5, cache_duration: int = 30) -> bool:
    """Smart filtering to reduce API calls"""
    # Only process high-confidence detections
    if conf < min_confidence:
        return False
    
    # Skip if we recently processed this track
    if track_id in ocr_cache:
        return False
    
    return True


def _iter_frames_from_path(path: str):
    cap = cv2.VideoCapture(path)
    if not cap or not cap.isOpened():
        raise RuntimeError(f"Failed to open input source: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def _iter_frames_from_folder(folder: str):
    import os
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".mp4", ".avi", ".mov", ".mkv"}
    for name in sorted(os.listdir(folder)):
        full = os.path.join(folder, name)
        if os.path.isdir(full):
            continue
        if os.path.splitext(name)[1].lower() in exts:
            if os.path.splitext(name)[1].lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                for f in _iter_frames_from_path(full):
                    yield f
            else:
                img = cv2.imread(full)
                if img is not None:
                    yield img


def _draw_annotations(frame, tracks, plate_texts):
    """Draw bounding boxes, labels, and track IDs on frame"""
    try:
        from ultralytics.utils.plotting import Annotator, colors
        annotator = Annotator(frame, line_width=2, font_size=12)
        
        for i, (track_id, cls, conf, box) in enumerate(tracks):
            x1, y1, x2, y2 = box
            plate_text = plate_texts[i] if i < len(plate_texts) else ""
            
            # Create label with track ID and plate text
            label = f"ID:{track_id}"
            if plate_text and plate_text != "None":
                label += f" | {plate_text}"
            label += f" | {conf:.2f}"
            
            # Draw bounding box and label
            annotator.box_label(box, label=label, color=colors(cls, True))
        
        return annotator.result()
    except ImportError:
        # Fallback to OpenCV if ultralytics plotting not available
        for i, (track_id, cls, conf, box) in enumerate(tracks):
            x1, y1, x2, y2 = box
            plate_text = plate_texts[i] if i < len(plate_texts) else ""
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Create label
            label = f"ID:{track_id}"
            if plate_text and plate_text != "None":
                label += f" | {plate_text}"
            label += f" | {conf:.2f}"
            
            # Draw label background and text
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame


def run_pipeline_cli(input_source: str, ocr_backend: str, save_path: str = "", show_window: bool = True, 
                    min_confidence: float = 0.5, cache_duration: int = 30) -> None:
    """
    Run ANPR pipeline with smart optimizations for cost reduction
    
    Args:
        min_confidence: Only process detections above this confidence
        cache_duration: How many frames to cache OCR results per track
    """
    cfg = load_config()
    det = YoloDetector(
        weights_path=cfg.get("yolo", {}).get("weights", "yolov8n.pt"),
        confidence_threshold=cfg.get("yolo", {}).get("confidence_threshold", 0.25),
        device=cfg.get("runtime", {}).get("device", "cpu"),
        tracker=cfg.get("yolo", {}).get("tracker", "bytetrack.yaml"),
    )
    ocr = _get_backend(ocr_backend, cfg)

    frame_iter = None
    if input_source.isdigit():
        cap = cv2.VideoCapture(int(input_source))
        if not cap or not cap.isOpened():
            raise RuntimeError("Failed to open camera index")
        def cam_iter():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        frame_iter = cam_iter()
    else:
        import os
        if os.path.isdir(input_source):
            frame_iter = _iter_frames_from_folder(input_source)
        else:
            frame_iter = _iter_frames_from_path(input_source)

    out_rows = []
    frame_no = 0
    api_calls = 0
    ocr_cache = {}  # track_id -> plate_text
    track_frames = {}  # track_id -> last_seen_frame
    
    print(f"Processing every frame with smart filtering")
    print(f"OCR backend: {ocr_backend}")
    print(f"Min confidence for OCR: {min_confidence}")
    print(f"Cache duration: {cache_duration} frames")
    
    for frame in frame_iter:
        frame_no += 1
        tracks = det.detect_and_track(frame)
        
        # Process OCR for each track with smart filtering
        plate_texts = []
        for track_id, cls, conf, box in tracks:
            # Check if we should process this track
            should_process = _should_process_track(track_id, conf, ocr_cache, min_confidence, cache_duration)
            
            if should_process:
                # New track or high-confidence detection - do OCR
                img_b64 = _bgr_crop_to_b64(frame, box)
                try:
                    text = ocr.extract_text(img_b64)
                    ocr_cache[track_id] = text or ""
                    track_frames[track_id] = frame_no
                    api_calls += 1
                    plate_texts.append(text or "")
                except Exception as e:
                    print(f"OCR error for track {track_id}: {e}")
                    plate_texts.append("")
            else:
                # Use cached result
                cached_text = ocr_cache.get(track_id, "")
                plate_texts.append(cached_text)
            
            # Clean up old cache entries
            if track_id in track_frames and frame_no - track_frames[track_id] > cache_duration:
                del ocr_cache[track_id]
                del track_frames[track_id]
            
            out_rows.append({
                "timestamp": now_iso(),
                "frame_no": frame_no,
                "track_id": track_id,
                "plate_text": plate_texts[-1],
                "confidence": conf,
                "x1": box[0],
                "y1": box[1],
                "x2": box[2],
                "y2": box[3],
            })

        # Draw annotations on frame
        annotated_frame = _draw_annotations(frame, tracks, plate_texts)

        if save_path and out_rows:
            save_any(save_path, out_rows)
            out_rows = []

        if show_window:
            try:
                cv2.imshow("ANPR", annotated_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            except Exception:
                pass
        
        # Print progress every 30 frames
        if frame_no % 30 == 0:
            print(f"Frame {frame_no}, API calls: {api_calls}, Cache size: {len(ocr_cache)}")

    print(f"\nProcessing complete!")
    print(f"Total frames processed: {frame_no}")
    print(f"Total API calls: {api_calls}")
    print(f"Average API calls per frame: {api_calls/frame_no:.2f}")
    print(f"Cost reduction: {((frame_no - api_calls) / frame_no * 100):.1f}% fewer API calls")

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
