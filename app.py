import argparse
import os

from src.gui import run_gui
from src.pipeline import run_pipeline_cli


def _get_local_models():
    """Get list of local model files from models folder"""
    models = []
    if os.path.exists("models"):
        for file in os.listdir("models"):
            if file.endswith((".pt", ".pth", ".onnx")):
                models.append(os.path.join("models", file))
    return models


def main():
    parser = argparse.ArgumentParser(description="ANPR: YOLO + OpenAI/Gemini (Optimized)")
    parser.add_argument("--gui", action="store_true", help="Launch GUI")
    parser.add_argument("--input", type=str, default="", help="Video file, folder, or camera index")
    parser.add_argument("--backend", type=str, default="openai", choices=["openai", "gemini", "dummy"], help="OCR backend")
    parser.add_argument("--save", type=str, default="", help="Save path (csv/json/sqlite)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model to use")
    parser.add_argument("--list-models", action="store_true", help="List available local models")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum confidence for OCR processing (0.0-1.0)")
    parser.add_argument("--cache-duration", type=int, default=30, help="Cache duration in frames (10-100)")
    args = parser.parse_args()

    if args.list_models:
        local_models = _get_local_models()
        if local_models:
            print("Available local models:")
            for model in local_models:
                print(f"  {model}")
        else:
            print("No local models found in \"models\" folder")
        return

    if args.gui:
        run_gui()
        return

    # Update config with selected model
    from src.utils import load_config
    import yaml
    cfg = load_config()
    cfg["yolo"]["weights"] = args.model
    with open("config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    run_pipeline_cli(input_source=args.input, ocr_backend=args.backend, save_path=args.save,
                    min_confidence=args.min_confidence, cache_duration=args.cache_duration)


if __name__ == "__main__":
    main()
