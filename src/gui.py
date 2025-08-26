
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import os

from .pipeline import run_pipeline_cli


def _get_local_models():
    """Get list of local model files from models folder"""
    models = []
    if os.path.exists("models"):
        for file in os.listdir("models"):
            if file.endswith((".pt", ".pth", ".onnx")):
                models.append(os.path.join("models", file))
    return models


def _start_cli(input_path: str, backend: str, save_path: str, model: str, min_conf: float, cache_dur: int) -> None:
    try:
        # Update config with selected model
        from .utils import load_config
        import yaml
        cfg = load_config()
        cfg["yolo"]["weights"] = model
        with open("config.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        
        run_pipeline_cli(input_source=input_path, ocr_backend=backend, save_path=save_path, 
                        min_confidence=min_conf, cache_duration=cache_dur)
    except Exception as e:
        messagebox.showerror("Error", str(e))


def run_gui() -> None:
    root = tk.Tk()
    root.title("ANPR - YOLO + OpenAI/Gemini (Optimized)")
    root.protocol("WM_DELETE_WINDOW", root.destroy)

    input_var = tk.StringVar()
    backend_var = tk.StringVar(value="dummy")
    save_var = tk.StringVar()
    model_var = tk.StringVar(value="yolov8n.pt")
    min_conf_var = tk.DoubleVar(value=0.5)
    cache_dur_var = tk.IntVar(value=30)

    frm = ttk.Frame(root, padding=12)
    frm.grid()

    ttk.Label(frm, text="Input (file or camera index)").grid(column=0, row=0, sticky="w")
    ttk.Entry(frm, textvariable=input_var, width=50).grid(column=1, row=0)
    ttk.Button(frm, text="Browse", command=lambda: input_var.set(filedialog.askopenfilename())).grid(column=2, row=0)

    ttk.Label(frm, text="YOLO Model").grid(column=0, row=1, sticky="w")
    
    # Get local models and add common ones
    local_models = _get_local_models()
    model_options = local_models + ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
    
    model_frame = ttk.Frame(frm)
    model_frame.grid(column=1, row=1, sticky="w")
    
    ttk.Combobox(model_frame, textvariable=model_var, values=model_options, width=30).grid(column=0, row=0, sticky="w")
    ttk.Button(model_frame, text="Browse", command=lambda: model_var.set(filedialog.askopenfilename(
        title="Select YOLO Model",
        filetypes=[("PyTorch models", "*.pt"), ("ONNX models", "*.onnx"), ("All files", "*.*")]
    ))).grid(column=1, row=0, padx=(5, 0))

    ttk.Label(frm, text="OCR Backend").grid(column=0, row=2, sticky="w")
    ttk.Combobox(frm, textvariable=backend_var, values=["openai", "gemini", "dummy"], width=20).grid(column=1, row=2, sticky="w")

    ttk.Label(frm, text="Save to (csv/json/sqlite)").grid(column=0, row=3, sticky="w")
    ttk.Entry(frm, textvariable=save_var, width=50).grid(column=1, row=3)
    ttk.Button(frm, text="Browse", command=lambda: save_var.set(filedialog.asksaveasfilename())).grid(column=2, row=3)

    # Optimization settings
    ttk.Label(frm, text="Min Confidence (0.0-1.0)").grid(column=0, row=4, sticky="w")
    min_conf_label = ttk.Label(frm, text=f"{min_conf_var.get():.2f}")
    min_conf_label.grid(column=2, row=4, sticky="w")
    
    def update_min_conf_label(val):
        rounded_val = round(float(val), 2)
        min_conf_var.set(rounded_val)
        min_conf_label.config(text=f"{rounded_val:.2f}")

    ttk.Scale(frm, from_=0.0, to=1.0, variable=min_conf_var, orient="horizontal", length=200, command=update_min_conf_label).grid(column=1, row=4, sticky="w")

    ttk.Label(frm, text="Cache Duration (frames)").grid(column=0, row=5, sticky="w")
    cache_dur_label = ttk.Label(frm, text=f"{cache_dur_var.get()}")
    cache_dur_label.grid(column=2, row=5, sticky="w")

    def update_cache_dur_label(val):
        rounded_val = int(float(val))
        cache_dur_var.set(rounded_val)
        cache_dur_label.config(text=f"{rounded_val}")

    ttk.Scale(frm, from_=10, to=100, variable=cache_dur_var, orient="horizontal", length=200, command=update_cache_dur_label).grid(column=1, row=5, sticky="w")

    ttk.Button(frm, text="Start", command=lambda: threading.Thread(target=_start_cli, 
                args=(input_var.get(), backend_var.get(), save_var.get(), model_var.get(), 
                      min_conf_var.get(), cache_dur_var.get()), daemon=True).start()).grid(column=0, row=6, pady=8)
    ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=6, pady=8)

    root.mainloop()
