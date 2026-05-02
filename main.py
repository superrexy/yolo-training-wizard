import os
import sys
import csv
import json
import platform
import shutil
import subprocess
import time
import zipfile
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
from rich.rule import Rule
from rich.text import Text
from rich import box
from rich.columns import Columns

console = Console()

# ─── Constants & Configuration ────────────────────────────────────────────────

CONFIG_DIR = Path.home() / ".yolo-wizard"
CONFIG_FILE = CONFIG_DIR / "config.json"

YOLO_MODELS = [
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
    "yolo26l.pt",
    "yolo26x.pt",
    "yolo12n.pt",
    "yolo12s.pt",
    "yolo12m.pt",
    "yolo12l.pt",
    "yolo12x.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
]

TASKS = ["detect", "segment", "classify", "pose", "obb"]

OUTPUT_DIR = Path("runs")
DATASETS_DIR = Path("datasets")

# ─── Training Presets (Phase 1.4) ─────────────────────────────────────────────

TRAINING_PRESETS = {
    "quick": {
        "label": "Quick Test",
        "description": "Fast validation that everything works (10 epochs, small images)",
        "epochs": 10,
        "imgsz": 320,
        "batch": -1,
        "patience": 5,
        "cos_lr": False,
        "close_mosaic": 5,
    },
    "balanced": {
        "label": "Balanced",
        "description": "Good quality with reasonable training time",
        "epochs": 100,
        "imgsz": 640,
        "batch": -1,
        "patience": 50,
        "cos_lr": True,
        "close_mosaic": 10,
    },
    "max_quality": {
        "label": "Maximum Quality",
        "description": "Best possible model — long training time",
        "epochs": 300,
        "imgsz": 640,
        "batch": -1,
        "patience": 100,
        "cos_lr": True,
        "close_mosaic": 20,
        "optimizer": "AdamW",
        "label_smoothing": 0.1,
    },
    "small_objects": {
        "label": "Small Objects",
        "description": "Optimized for detecting small objects (high resolution)",
        "epochs": 200,
        "imgsz": 1280,
        "batch": -1,
        "patience": 80,
        "cos_lr": True,
        "close_mosaic": 15,
        "multi_scale": True,
    },
    "finetune": {
        "label": "Fine-Tune",
        "description": "Fine-tune a pretrained model on your data (frozen backbone first)",
        "epochs": 50,
        "imgsz": 640,
        "batch": -1,
        "patience": 20,
        "cos_lr": True,
        "close_mosaic": 10,
        "freeze": 10,
        "lr0": 0.001,
    },
    "custom": {
        "label": "Custom",
        "description": "Configure everything manually",
    },
}

# ─── Augmentation Presets (Phase 2.2) ─────────────────────────────────────────

AUGMENTATION_PRESETS = {
    "none": {
        "label": "None",
        "description": "No augmentation (for debugging or very clean datasets)",
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
    },
    "light": {
        "label": "Light",
        "description": "Gentle augmentation for small/clean datasets",
        "hsv_h": 0.01,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "degrees": 0.0,
        "translate": 0.05,
        "scale": 0.3,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.5,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
    },
    "medium": {
        "label": "Medium (Default)",
        "description": "Ultralytics default augmentation — good for most cases",
    },
    "heavy": {
        "label": "Heavy",
        "description": "Aggressive augmentation for large models or overfitting prevention",
        "hsv_h": 0.02,
        "hsv_s": 0.9,
        "hsv_v": 0.5,
        "degrees": 10.0,
        "translate": 0.2,
        "scale": 0.7,
        "shear": 5.0,
        "perspective": 0.001,
        "flipud": 0.1,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.3,
        "copy_paste": 0.1,
        "erasing": 0.3,
    },
    "custom": {
        "label": "Custom",
        "description": "Configure each augmentation parameter individually",
    },
}

# ─── YOLO26 Model-Size Aware Defaults (Phase 2.3) ────────────────────────────

YOLO26_DEFAULTS = {
    "n": {"lr0": 0.0054, "lrf": 0.0495, "weight_decay": 0.00064, "momentum": 0.947},
    "s": {"lr0": 0.00038, "lrf": 0.882, "weight_decay": 0.00027, "momentum": 0.948},
    "m": {"lr0": 0.00038, "lrf": 0.882, "weight_decay": 0.00027, "momentum": 0.948},
    "l": {"lr0": 0.00038, "lrf": 0.882, "weight_decay": 0.00027, "momentum": 0.948},
    "x": {"lr0": 0.00038, "lrf": 0.882, "weight_decay": 0.00027, "momentum": 0.948},
}

# ─── Export Formats (Phase 3.2) ───────────────────────────────────────────────

EXPORT_FORMATS = {
    "onnx": "ONNX (cross-platform, recommended)",
    "engine": "TensorRT (NVIDIA GPU inference, fastest)",
    "coreml": "CoreML (Apple devices)",
    "tflite": "TFLite (mobile/edge devices)",
    "openvino": "OpenVINO (Intel hardware)",
    "ncnn": "NCNN (mobile, lightweight)",
    "torchscript": "TorchScript (PyTorch deployment)",
}


# ─── Config Helpers ───────────────────────────────────────────────────────────


def _load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_config(config: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def _mask_key(key: str) -> str:
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"


def _prompt_api_key() -> str:
    saved = _load_config()
    saved_key = saved.get("roboflow_api_key")

    if saved_key:
        masked = _mask_key(saved_key)
        console.print(f"[dim]Saved API key found:[/] [bold]{masked}[/]")
        console.print()

        choice = Prompt.ask(
            "[yellow]Use saved API key?[/]",
            choices=["yes", "new"],
            default="yes",
        )

        if choice == "yes":
            console.print("[green]Using saved API key.[/]")
            return saved_key

    new_key = Prompt.ask("[yellow]Enter Roboflow API Key[/]", password=True)

    if not new_key.strip():
        console.print("[red]API key cannot be empty.[/]")
        sys.exit(1)

    new_key = new_key.strip()

    save_it = Confirm.ask("[yellow]Save this API key for future use?[/]", default=True)
    if save_it:
        config = _load_config()
        config["roboflow_api_key"] = new_key
        _save_config(config)
        console.print(f"[green]API key saved to {CONFIG_FILE}[/]")

    return new_key


# ─── Dataset Helpers ──────────────────────────────────────────────────────────


def _find_data_yaml(dataset_path: str) -> str | None:
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f == "data.yaml":
                return os.path.join(root, f)
    return None


def _get_classes_from_yaml(yaml_path: str) -> list[str]:
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        if isinstance(names, dict):
            return list(names.values())
        return names
    except Exception:
        return []


def _get_dataset_stats(yaml_path: str) -> dict:
    """Get dataset statistics for health checks."""
    stats = {"train_images": 0, "val_images": 0, "test_images": 0}
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        base_dir = Path(yaml_path).parent

        for split in ["train", "val", "test"]:
            split_path = data.get(split, "")
            if split_path:
                full_path = base_dir / split_path
                if full_path.exists():
                    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
                    count = sum(
                        1 for f in full_path.iterdir()
                        if f.is_file() and f.suffix.lower() in img_extensions
                    )
                    stats[f"{split}_images"] = count
    except Exception:
        pass
    return stats


# ─── Device & System Helpers ──────────────────────────────────────────────────


def _format_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _detect_device() -> tuple[str, str]:
    """
    Auto-detect best available training device.
    Returns (device_value, description) where device_value is what Ultralytics expects.
    Priority: CUDA (all GPUs) -> MPS (Apple Silicon) -> CPU
    """
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
                gpu_count = len(gpu_lines)
                gpu_names = [line.split(",")[1].strip() for line in gpu_lines]

                if gpu_count > 1:
                    device_value = ",".join(str(i) for i in range(gpu_count))
                    desc = f"CUDA multi-GPU ({gpu_count}x: {', '.join(gpu_names)})"
                else:
                    device_value = "0"
                    desc = f"CUDA ({gpu_names[0]})"
                return device_value, desc
        except (subprocess.TimeoutExpired, OSError):
            pass

    if platform.system() == "Darwin":
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps", "MPS (Apple Silicon)"
        except (ImportError, AttributeError):
            return "mps", "MPS (Apple Silicon - torch check skipped)"

    return "cpu", "CPU (no GPU detected)"


def _check_disk_space() -> float:
    """Return free disk space in GB."""
    usage = shutil.disk_usage(".")
    return usage.free / (1024**3)


def _check_resumable_runs() -> list[dict]:
    """Scan runs/ for interrupted training sessions."""
    resumable = []
    runs_dir = OUTPUT_DIR

    if not runs_dir.exists():
        return resumable

    for last_pt in runs_dir.rglob("weights/last.pt"):
        train_dir = last_pt.parent.parent
        results_csv = train_dir / "results.csv"
        args_yaml = train_dir / "args.yaml"

        if not results_csv.exists() or not args_yaml.exists():
            continue

        try:
            with open(results_csv) as f:
                completed_epochs = sum(1 for _ in csv.reader(f)) - 1  # minus header

            with open(args_yaml) as f:
                args = yaml.safe_load(f)

            total_epochs = args.get("epochs", 0)

            if completed_epochs > 0 and completed_epochs < total_epochs:
                resumable.append({
                    "path": str(last_pt),
                    "dir": str(train_dir),
                    "model": args.get("model", "unknown"),
                    "completed": completed_epochs,
                    "total": total_epochs,
                    "data": args.get("data", ""),
                })
        except Exception:
            continue

    return resumable


def _get_model_size_key(model_name: str) -> str:
    """Extract size key (n/s/m/l/x) from model name."""
    for size in ["n", "s", "m", "l", "x"]:
        if model_name.rstrip(".pt").endswith(size):
            return size
    return "s"  # default


def _is_yolo26(model_name: str) -> bool:
    """Check if model is YOLO26 variant."""
    return "yolo26" in model_name.lower()


def _parse_batch_input(batch_str: str, device: str) -> int | float:
    """Parse batch size input: 'auto' -> -1, 'auto-70' -> 0.7, integer -> int."""
    batch_str = batch_str.strip().lower()

    if batch_str == "auto":
        return -1
    if batch_str.startswith("auto-"):
        try:
            pct = int(batch_str.split("-")[1])
            return pct / 100.0
        except (ValueError, IndexError):
            return -1

    try:
        val = int(batch_str)
        return val
    except ValueError:
        return -1


# ─── Pre-Training Health Checks (Phase 4.2) ──────────────────────────────────


def _run_health_checks(config: dict, dataset_info: dict) -> list[tuple[str, str]]:
    """
    Run pre-training health checks.
    Returns list of (severity, message) tuples.
    severity: 'info', 'warning', 'error'
    """
    checks = []

    # 1. Disk space check
    free_gb = _check_disk_space()
    if free_gb < 5:
        checks.append(("error", f"Very low disk space: {free_gb:.1f}GB free. Training may fail."))
    elif free_gb < 15:
        checks.append(("warning", f"Low disk space: {free_gb:.1f}GB free. Consider freeing space."))
    else:
        checks.append(("info", f"Disk space: {free_gb:.1f}GB free"))

    # 2. Dataset split verification
    if dataset_info.get("data_yaml"):
        stats = _get_dataset_stats(dataset_info["data_yaml"])
        train_count = stats.get("train_images", 0)
        val_count = stats.get("val_images", 0)

        if train_count > 0:
            checks.append(("info", f"Training images: {train_count}"))
        else:
            checks.append(("warning", "Could not count training images"))

        if val_count > 0:
            checks.append(("info", f"Validation images: {val_count}"))
        else:
            checks.append(("warning", "Could not count validation images"))

        # Small dataset warning
        if 0 < train_count < 100:
            checks.append(("warning", f"Very small dataset ({train_count} images). Consider using 'finetune' preset with frozen backbone."))
        elif 0 < train_count < 500:
            checks.append(("info", f"Small dataset. Heavy augmentation recommended."))

    # 3. Class count check
    num_classes = len(dataset_info.get("classes", []))
    if num_classes > 0:
        checks.append(("info", f"Classes: {num_classes}"))
        if num_classes > 80:
            checks.append(("warning", f"Many classes ({num_classes}). Consider using a larger model (m/l/x)."))

    # 4. Batch size + device compatibility
    batch = config.get("batch", 16)
    device = config.get("device", "cpu")
    if batch == -1 and device == "cpu":
        checks.append(("warning", "Auto-batch requires GPU. Falling back to batch=16 for CPU."))

    # 5. Image size check
    imgsz = config.get("imgsz", 640)
    if imgsz > 640 and batch != -1 and batch > 8:
        checks.append(("warning", f"Large image size ({imgsz}) with batch={batch} may cause OOM. Consider batch=auto."))

    # 6. Epochs vs patience
    epochs = config.get("epochs", 100)
    patience = config.get("patience", 50)
    if patience >= epochs:
        checks.append(("warning", f"Patience ({patience}) >= epochs ({epochs}). Early stopping will never trigger."))

    return checks


# ─── Step 0: Resume Check ────────────────────────────────────────────────────


def step_check_resume() -> dict | None:
    """Check for resumable training runs. Returns resume config or None."""
    resumable = _check_resumable_runs()

    if not resumable:
        return None

    console.print(Rule("[bold yellow]Interrupted Training Detected[/]"))
    console.print()

    table = Table(title="Resumable Runs", box=box.ROUNDED)
    table.add_column("#", style="dim")
    table.add_column("Directory", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Progress", style="yellow")

    for i, run in enumerate(resumable, 1):
        progress_str = f"{run['completed']}/{run['total']} epochs ({run['completed']*100//run['total']}%)"
        table.add_row(str(i), Path(run["dir"]).name, run["model"], progress_str)

    console.print(table)
    console.print()

    resume_choice = Prompt.ask(
        "[yellow]Resume a previous run?[/]",
        choices=["skip"] + [str(i) for i in range(1, len(resumable) + 1)],
        default="skip",
    )

    if resume_choice == "skip":
        return None

    selected = resumable[int(resume_choice) - 1]
    console.print(f"[green]Resuming training from {selected['path']}[/]")
    console.print()

    return selected


# ─── Step 1: Download Dataset ─────────────────────────────────────────────────


def step_download_dataset() -> dict:
    console.print(Rule("[bold cyan]Step 1: Download Dataset from Roboflow[/]"))
    console.print()

    api_key = _prompt_api_key()
    console.print()
    workspace = Prompt.ask("[yellow]Workspace name[/]")
    project_name = Prompt.ask("[yellow]Project name[/]")
    version_number = IntPrompt.ask("[yellow]Dataset version[/]", default=1)
    dataset_format = Prompt.ask(
        "[yellow]Export format[/]",
        default="yolo26",
        choices=["yolo26", "yolov12", "yolov11", "yolov8", "yolov5", "yolov7", "yolov9", "coco", "voc"],
    )

    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Connecting to Roboflow...", total=None)

        from roboflow import Roboflow

        rf = Roboflow(api_key=api_key)

        progress.update(task, description="Fetching project...")
        project = rf.workspace(workspace).project(project_name)

        progress.update(task, description="Fetching dataset version...")
        version = project.version(version_number)

        progress.update(task, description=f"Downloading dataset ({dataset_format})...")
        dataset_location = str(DATASETS_DIR / f"{project_name}-{version_number}")
        version.download(dataset_format, location=dataset_location)

        progress.update(task, description="[green]Download complete!")

    data_yaml = _find_data_yaml(dataset_location)
    classes = _get_classes_from_yaml(data_yaml) if data_yaml else []

    info = {
        "workspace": workspace,
        "project": project_name,
        "version": version_number,
        "format": dataset_format,
        "path": dataset_location,
        "data_yaml": data_yaml,
        "classes": classes,
    }

    console.print()
    table = Table(title="Dataset Info", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Project", f"{workspace}/{project_name}")
    table.add_row("Version", str(version_number))
    table.add_row("Format", dataset_format)
    table.add_row("Location", dataset_location)
    if classes:
        table.add_row("Classes", ", ".join(classes[:20]) + ("..." if len(classes) > 20 else ""))
        table.add_row("Num Classes", str(len(classes)))

    # Show dataset stats
    if data_yaml:
        stats = _get_dataset_stats(data_yaml)
        if stats["train_images"]:
            table.add_row("Train Images", str(stats["train_images"]))
        if stats["val_images"]:
            table.add_row("Val Images", str(stats["val_images"]))

    console.print(table)
    console.print()

    return info


# ─── Step 2: Configure Training ──────────────────────────────────────────────


def _select_model() -> str:
    """Interactive model selection with grouping."""
    console.print("[bold]Available models:[/]")
    console.print()

    # Group models by generation
    groups = {
        "YOLO26 (Latest)": [m for m in YOLO_MODELS if "yolo26" in m],
        "YOLO12": [m for m in YOLO_MODELS if "yolo12" in m],
        "YOLO11": [m for m in YOLO_MODELS if "yolo11" in m],
        "YOLOv8": [m for m in YOLO_MODELS if "yolov8" in m],
    }

    idx = 1
    model_map = {}
    for group_name, models in groups.items():
        console.print(f"  [bold dim]{group_name}:[/]")
        for model in models:
            size = _get_model_size_key(model)
            size_desc = {"n": "nano", "s": "small", "m": "medium", "l": "large", "x": "xlarge"}
            console.print(f"    [dim]{idx:2d}.[/] {model} [dim]({size_desc.get(size, size)})[/]")
            model_map[idx] = model
            idx += 1
        console.print()

    model_idx = IntPrompt.ask(
        "[yellow]Select model number[/]",
        default=1,
        choices=[str(i) for i in range(1, idx)],
    )
    return model_map[model_idx]


def _select_preset() -> dict:
    """Interactive training preset selection."""
    console.print("[bold]Training Presets:[/]")
    console.print()

    preset_keys = list(TRAINING_PRESETS.keys())
    for i, key in enumerate(preset_keys, 1):
        preset = TRAINING_PRESETS[key]
        label = preset["label"]
        desc = preset["description"]
        console.print(f"  [dim]{i}.[/] [bold]{label}[/] — [dim]{desc}[/]")

    console.print()

    preset_idx = IntPrompt.ask(
        "[yellow]Select preset[/]",
        default=2,  # Balanced
        choices=[str(i) for i in range(1, len(preset_keys) + 1)],
    )

    selected_key = preset_keys[preset_idx - 1]
    selected = TRAINING_PRESETS[selected_key].copy()
    selected.pop("label", None)
    selected.pop("description", None)

    if selected_key != "custom":
        console.print(f"[green]Using preset: {TRAINING_PRESETS[selected_key]['label']}[/]")

    return {"_preset_key": selected_key, **selected}


def _configure_augmentation() -> dict:
    """Interactive augmentation configuration."""
    console.print()
    console.print("[bold]Augmentation Level:[/]")
    console.print()

    aug_keys = list(AUGMENTATION_PRESETS.keys())
    for i, key in enumerate(aug_keys, 1):
        preset = AUGMENTATION_PRESETS[key]
        console.print(f"  [dim]{i}.[/] [bold]{preset['label']}[/] — [dim]{preset['description']}[/]")

    console.print()

    aug_idx = IntPrompt.ask(
        "[yellow]Select augmentation level[/]",
        default=3,  # Medium
        choices=[str(i) for i in range(1, len(aug_keys) + 1)],
    )

    selected_key = aug_keys[aug_idx - 1]
    selected = AUGMENTATION_PRESETS[selected_key].copy()
    selected.pop("label", None)
    selected.pop("description", None)

    if selected_key == "custom":
        # Prompt each augmentation parameter
        selected = {
            "hsv_h": FloatPrompt.ask("[yellow]  HSV-Hue[/]", default=0.015),
            "hsv_s": FloatPrompt.ask("[yellow]  HSV-Saturation[/]", default=0.7),
            "hsv_v": FloatPrompt.ask("[yellow]  HSV-Value[/]", default=0.4),
            "degrees": FloatPrompt.ask("[yellow]  Rotation degrees[/]", default=0.0),
            "translate": FloatPrompt.ask("[yellow]  Translation[/]", default=0.1),
            "scale": FloatPrompt.ask("[yellow]  Scale[/]", default=0.5),
            "shear": FloatPrompt.ask("[yellow]  Shear[/]", default=0.0),
            "perspective": FloatPrompt.ask("[yellow]  Perspective[/]", default=0.0),
            "flipud": FloatPrompt.ask("[yellow]  Flip up-down prob[/]", default=0.0),
            "fliplr": FloatPrompt.ask("[yellow]  Flip left-right prob[/]", default=0.5),
            "mosaic": FloatPrompt.ask("[yellow]  Mosaic prob[/]", default=1.0),
            "mixup": FloatPrompt.ask("[yellow]  Mixup prob[/]", default=0.0),
            "copy_paste": FloatPrompt.ask("[yellow]  Copy-paste prob[/]", default=0.0),
            "erasing": FloatPrompt.ask("[yellow]  Random erasing prob[/]", default=0.0),
        }

    return selected


def _configure_optimizer_lr(model_name: str) -> dict:
    """Configure optimizer and learning rate parameters."""
    console.print()
    console.print("[bold]Optimizer & Learning Rate:[/]")
    console.print()

    # Determine smart defaults based on model
    is_26 = _is_yolo26(model_name)
    size_key = _get_model_size_key(model_name)

    if is_26 and size_key in YOLO26_DEFAULTS:
        defaults = YOLO26_DEFAULTS[size_key]
        console.print(f"[dim]  Using YOLO26-{size_key} optimized defaults[/]")
    else:
        defaults = {"lr0": 0.01, "lrf": 0.01, "weight_decay": 0.0005, "momentum": 0.937}

    optimizer = Prompt.ask(
        "[yellow]  Optimizer[/]",
        default="auto",
        choices=["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam"],
    )
    cos_lr = Confirm.ask("[yellow]  Cosine LR scheduler?[/]", default=True)
    lr0 = FloatPrompt.ask("[yellow]  Initial learning rate[/]", default=defaults["lr0"])
    lrf = FloatPrompt.ask("[yellow]  Final LR ratio (lrf)[/]", default=defaults["lrf"])
    momentum = FloatPrompt.ask("[yellow]  Momentum[/]", default=defaults["momentum"])
    weight_decay = FloatPrompt.ask("[yellow]  Weight decay[/]", default=defaults["weight_decay"])
    warmup_epochs = FloatPrompt.ask("[yellow]  Warmup epochs[/]", default=3.0)
    warmup_momentum = FloatPrompt.ask("[yellow]  Warmup momentum[/]", default=0.8)
    warmup_bias_lr = FloatPrompt.ask("[yellow]  Warmup bias LR[/]", default=0.1)
    nbs = IntPrompt.ask("[yellow]  Nominal batch size (nbs)[/]", default=64)

    return {
        "optimizer": optimizer,
        "cos_lr": cos_lr,
        "lr0": lr0,
        "lrf": lrf,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "warmup_momentum": warmup_momentum,
        "warmup_bias_lr": warmup_bias_lr,
        "nbs": nbs,
    }


def _configure_caching_performance(device: str) -> dict:
    """Configure caching and performance options."""
    console.print()
    console.print("[bold]Caching & Performance:[/]")
    console.print()

    cache_choice = Prompt.ask(
        "[yellow]  Dataset caching[/]",
        default="False",
        choices=["False", "ram", "disk"],
    )
    cache = cache_choice if cache_choice != "False" else False

    amp = Confirm.ask("[yellow]  Mixed precision (AMP)?[/]", default=True)
    multi_scale = Confirm.ask("[yellow]  Multi-scale training?[/]", default=False)
    rect = Confirm.ask("[yellow]  Rectangular training?[/]", default=False)

    result = {"cache": cache, "amp": amp}
    if multi_scale:
        result["multi_scale"] = True
    if rect:
        result["rect"] = True

    return result


def _configure_reproducibility() -> dict:
    """Configure reproducibility options."""
    console.print()
    console.print("[bold]Reproducibility:[/]")
    console.print()

    seed = IntPrompt.ask("[yellow]  Random seed[/]", default=0)
    deterministic = Confirm.ask("[yellow]  Deterministic mode?[/]", default=True)

    return {"seed": seed, "deterministic": deterministic}


def step_configure_training(dataset_info: dict) -> dict:
    console.print(Rule("[bold cyan]Step 2: Configure Training Parameters[/]"))
    console.print()

    # Model selection
    model = _select_model()
    task = Prompt.ask("[yellow]Task[/]", default="detect", choices=TASKS)

    console.print()

    # Preset selection
    preset_config = _select_preset()
    preset_key = preset_config.pop("_preset_key")

    # Device detection
    detected_device, device_desc = _detect_device()
    console.print()
    console.print(f"[bold]Detected device:[/] [green]{device_desc}[/]")
    override_device = Confirm.ask("[yellow]Use detected device?[/]", default=True)
    device = detected_device if override_device else Prompt.ask(
        "[yellow]Enter device manually[/]", default=detected_device
    )

    if preset_key == "custom":
        # Full manual configuration
        console.print()
        console.print("[bold]Training Parameters:[/]")
        epochs = IntPrompt.ask("[yellow]  Epochs[/]", default=100)

        # Auto batch support
        console.print("[dim]  Batch: 'auto' = auto-detect optimal, 'auto-70' = use 70% GPU, or integer[/]")
        batch_str = Prompt.ask("[yellow]  Batch size[/]", default="auto")
        batch = _parse_batch_input(batch_str, device)

        # Fallback for CPU
        if batch == -1 and device == "cpu":
            console.print("[yellow]  Auto-batch requires GPU. Using batch=16.[/]")
            batch = 16

        img_size = IntPrompt.ask("[yellow]  Image size[/]", default=640)
        lr0 = FloatPrompt.ask("[yellow]  Initial learning rate[/]", default=0.01)
        patience = IntPrompt.ask("[yellow]  Early stopping patience[/]", default=50)
        workers = IntPrompt.ask("[yellow]  Dataloader workers[/]", default=8)
        close_mosaic = IntPrompt.ask("[yellow]  Close mosaic (last N epochs)[/]", default=10)

        config = {
            "epochs": epochs,
            "batch": batch,
            "imgsz": img_size,
            "lr0": lr0,
            "patience": patience,
            "workers": workers,
            "close_mosaic": close_mosaic,
        }
    else:
        # Use preset values with optional overrides
        config = preset_config.copy()

        # Fix auto-batch for CPU
        if config.get("batch") == -1 and device == "cpu":
            console.print("[yellow]Auto-batch requires GPU. Using batch=16.[/]")
            config["batch"] = 16

        # Allow overriding key preset values
        console.print()
        if Confirm.ask("[yellow]Override any preset values?[/]", default=False):
            config["epochs"] = IntPrompt.ask("[yellow]  Epochs[/]", default=config.get("epochs", 100))
            config["imgsz"] = IntPrompt.ask("[yellow]  Image size[/]", default=config.get("imgsz", 640))
            config["patience"] = IntPrompt.ask("[yellow]  Patience[/]", default=config.get("patience", 50))

        # Set workers (not in presets)
        config.setdefault("workers", 8)

    # Output naming
    console.print()
    project_name = Prompt.ask(
        "[yellow]Output project name[/]",
        default=f"train-{dataset_info['project']}",
    )
    run_name = Prompt.ask(
        "[yellow]Run name[/]",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    # Advanced options (categorized)
    console.print()
    use_advanced = Confirm.ask("[yellow]Configure advanced options?[/]", default=False)

    advanced = {}
    if use_advanced:
        console.print()
        console.print("[bold]Advanced Configuration Categories:[/]")
        console.print("  [dim]1.[/] Optimizer & Learning Rate")
        console.print("  [dim]2.[/] Augmentation")
        console.print("  [dim]3.[/] Caching & Performance")
        console.print("  [dim]4.[/] Reproducibility")
        console.print("  [dim]5.[/] All of the above")
        console.print()

        adv_choice = Prompt.ask(
            "[yellow]Select categories (comma-separated)[/]",
            default="5",
        )
        categories = [c.strip() for c in adv_choice.split(",")]

        if "1" in categories or "5" in categories:
            advanced.update(_configure_optimizer_lr(model))

        if "2" in categories or "5" in categories:
            aug_params = _configure_augmentation()
            advanced.update(aug_params)

        if "3" in categories or "5" in categories:
            advanced.update(_configure_caching_performance(device))

        if "4" in categories or "5" in categories:
            advanced.update(_configure_reproducibility())

        # Additional advanced params
        console.print()
        console.print("[bold]Additional Options:[/]")
        freeze = IntPrompt.ask("[yellow]  Freeze layers (0=none)[/]", default=config.get("freeze", 0))
        dropout = FloatPrompt.ask("[yellow]  Dropout[/]", default=0.0)
        label_smoothing = FloatPrompt.ask("[yellow]  Label smoothing[/]", default=config.get("label_smoothing", 0.0))

        if freeze > 0:
            advanced["freeze"] = freeze
        if dropout > 0:
            advanced["dropout"] = dropout
        if label_smoothing > 0:
            advanced["label_smoothing"] = label_smoothing

    # ─── Phase 1.1: Smart Defaults (always applied silently) ──────────────────
    smart_defaults = {
        "amp": True,
        "plots": True,
        "val": True,
        "exist_ok": False,
        "verbose": True,
    }

    # Apply close_mosaic if not already set
    if "close_mosaic" not in config and "close_mosaic" not in advanced:
        smart_defaults["close_mosaic"] = 10

    # YOLO26 model-size-aware defaults (Phase 2.3)
    if _is_yolo26(model) and "lr0" not in advanced:
        size_key = _get_model_size_key(model)
        if size_key in YOLO26_DEFAULTS:
            yolo26_params = YOLO26_DEFAULTS[size_key]
            # Only apply if user hasn't set these
            for param, value in yolo26_params.items():
                if param not in config and param not in advanced:
                    smart_defaults[param] = value
            console.print(f"[dim]Applied YOLO26-{size_key} optimized hyperparameters[/]")

    # Save period for long runs (Phase 4.4)
    epochs = config.get("epochs", 100)
    if epochs > 100:
        save_period = max(epochs // 10, 10)
        smart_defaults["save_period"] = save_period

    # Build final config
    final_config = {
        "model": model,
        "task": task,
        "data": dataset_info["data_yaml"],
        "device": device,
        "project": str(OUTPUT_DIR / project_name),
        "name": run_name,
        **smart_defaults,
        **config,
        **advanced,
    }

    final_config = {k: v for k, v in final_config.items() if v is not None}

    # Display final configuration
    console.print()
    table = Table(title="Training Configuration", box=box.ROUNDED)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Source", style="dim")

    for key, value in sorted(final_config.items()):
        if key in smart_defaults and key not in config and key not in advanced:
            source = "auto"
        elif key in advanced:
            source = "advanced"
        elif key in config:
            source = "preset" if preset_key != "custom" else "manual"
        else:
            source = ""
        table.add_row(key, str(value), source)

    console.print(table)
    console.print()

    # Save config as YAML (Phase 4.1)
    save_yaml = Confirm.ask("[yellow]Save this configuration as YAML for reuse?[/]", default=False)
    if save_yaml:
        config_name = Prompt.ask("[yellow]Config filename[/]", default=f"config-{run_name}")
        yaml_path = Path(f"{config_name}.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(final_config, f, default_flow_style=False, sort_keys=True)
        console.print(f"[green]Configuration saved to {yaml_path}[/]")
        console.print()

    if not Confirm.ask("[yellow]Proceed with training?[/]", default=True):
        console.print("[red]Training cancelled by user.[/]")
        sys.exit(0)

    return final_config


# ─── Step 2.5: Pre-Training Health Checks (Phase 4.2) ────────────────────────


def step_health_checks(config: dict, dataset_info: dict) -> None:
    console.print(Rule("[bold cyan]Pre-Training Health Checks[/]"))
    console.print()

    checks = _run_health_checks(config, dataset_info)

    has_issues = False
    for severity, message in checks:
        if severity == "error":
            console.print(f"  [bold red]ERROR:[/] {message}")
            has_issues = True
        elif severity == "warning":
            console.print(f"  [yellow]WARNING:[/] {message}")
            has_issues = True
        else:
            console.print(f"  [green]OK:[/] {message}")

    console.print()

    if has_issues:
        if not Confirm.ask("[yellow]Issues detected. Continue anyway?[/]", default=True):
            console.print("[red]Training cancelled.[/]")
            sys.exit(0)
    else:
        console.print("[green]All checks passed![/]")
        console.print()


# ─── Step 3: Train Model ─────────────────────────────────────────────────────


def step_train(config: dict, resume_info: dict | None = None) -> dict:
    console.print(Rule("[bold cyan]Step 3: Training Model[/]"))
    console.print()

    from ultralytics import YOLO

    start_time = time.time()

    if resume_info:
        # Resume interrupted training
        console.print(f"[bold]Resuming from:[/] {resume_info['path']}")
        console.print(f"[dim]Completed: {resume_info['completed']}/{resume_info['total']} epochs[/]")
        model = YOLO(resume_info["path"])
        console.print("[bold]Resuming training...[/]")
        console.print()
        results = model.train(resume=True)
    else:
        # Fresh training
        console.print(f"[bold]Loading model:[/] {config['model']}")
        model = YOLO(config["model"])

        console.print("[bold]Starting training...[/]")
        console.print(f"[dim]Output directory: {config['project']}/{config['name']}[/]")
        console.print()

        # Build train args (exclude non-training keys)
        exclude_keys = {"task"}
        train_args = {k: v for k, v in config.items() if k not in exclude_keys}
        results = model.train(**train_args)

    # Use actual save directory from YOLO results (handles suffix increments like train2, train3)
    train_dir = Path(str(results.save_dir))

    elapsed = time.time() - start_time

    console.print()
    console.print("[bold green]Training complete![/]")
    console.print(f"[dim]Results saved to: {train_dir}[/]")
    console.print(f"[dim]Total time: {elapsed/60:.1f} minutes[/]")
    console.print()

    return {
        "results": results,
        "train_dir": str(train_dir),
        "model_path": str(train_dir / "weights" / "best.pt"),
        "elapsed_seconds": elapsed,
    }


# ─── Step 4: Validate Model (Phase 3.1) ──────────────────────────────────────


def step_validate(train_info: dict, config: dict) -> dict | None:
    console.print(Rule("[bold cyan]Step 4: Model Validation[/]"))
    console.print()

    run_val = Confirm.ask("[yellow]Run validation on best model?[/]", default=True)
    if not run_val:
        return None

    from ultralytics import YOLO

    console.print(f"[bold]Loading best model:[/] {train_info['model_path']}")
    model = YOLO(train_info["model_path"])

    console.print("[bold]Running validation...[/]")
    console.print()

    val_args = {
        "data": config.get("data"),
        "imgsz": config.get("imgsz", 640),
        "batch": config.get("batch", 16) if config.get("batch", 16) != -1 else 16,
        "device": config.get("device", "cpu"),
        "plots": True,
        "verbose": True,
    }

    metrics = model.val(**val_args)

    # Display validation results
    console.print()
    val_table = Table(title="Validation Results", box=box.ROUNDED)
    val_table.add_column("Metric", style="cyan")
    val_table.add_column("Value", style="green")

    try:
        if hasattr(metrics, "box"):
            val_table.add_row("mAP50", f"{metrics.box.map50:.4f}")
            val_table.add_row("mAP50-95", f"{metrics.box.map:.4f}")
            val_table.add_row("Precision", f"{metrics.box.mp:.4f}")
            val_table.add_row("Recall", f"{metrics.box.mr:.4f}")

            # Per-class mAP (Phase 3.3)
            if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
                console.print(val_table)
                console.print()

                class_table = Table(title="Per-Class mAP50-95", box=box.ROUNDED)
                class_table.add_column("Class", style="cyan")
                class_table.add_column("mAP", style="green")

                names = metrics.names if hasattr(metrics, "names") else {}
                for i, map_val in enumerate(metrics.box.maps):
                    class_name = names.get(i, f"class_{i}")
                    class_table.add_row(str(class_name), f"{map_val:.4f}")

                console.print(class_table)
            else:
                console.print(val_table)
        else:
            val_table.add_row("Results", str(metrics))
            console.print(val_table)
    except Exception as e:
        console.print(f"[yellow]Could not parse validation metrics: {e}[/]")

    console.print()
    return metrics


# ─── Step 5: Training Summary (Enhanced - Phase 3.3) ──────────────────────────


def _find_best_epoch(csv_path: Path) -> dict | None:
    """Find the epoch with best mAP50-95."""
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return None

        # Look for mAP column
        map_col = None
        for col in rows[0].keys():
            col_stripped = col.strip()
            if "metrics/mAP50-95" in col_stripped or "mAP_0.5:0.95" in col_stripped:
                map_col = col
                break

        if not map_col:
            return None

        best_row = max(rows, key=lambda r: float(r.get(map_col, "0").strip() or "0"))
        return best_row

    except Exception:
        return None


def _display_metrics_from_csv(csv_path: Path) -> None:
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return

        # Show last epoch metrics
        last_row = rows[-1]
        metrics_table = Table(title="Final Epoch Metrics", box=box.ROUNDED)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")

        for key, value in last_row.items():
            key = key.strip()
            try:
                val = float(value.strip())
                metrics_table.add_row(key, f"{val:.5f}")
            except (ValueError, AttributeError):
                metrics_table.add_row(key, str(value).strip())

        console.print(metrics_table)
        console.print()

        # Show best epoch (Phase 3.3)
        best_row = _find_best_epoch(csv_path)
        if best_row and best_row != last_row:
            best_table = Table(title="Best Epoch Metrics (by mAP50-95)", box=box.ROUNDED)
            best_table.add_column("Metric", style="cyan")
            best_table.add_column("Value", style="bold green")

            for key, value in best_row.items():
                key = key.strip()
                try:
                    val = float(value.strip())
                    best_table.add_row(key, f"{val:.5f}")
                except (ValueError, AttributeError):
                    best_table.add_row(key, str(value).strip())

            console.print(best_table)
            console.print()

    except Exception as e:
        console.print(f"[yellow]Could not parse results.csv: {e}[/]")


def step_summary(train_info: dict, config: dict, dataset_info: dict) -> None:
    console.print(Rule("[bold cyan]Step 5: Training Summary[/]"))
    console.print()

    train_dir = Path(train_info["train_dir"])

    # Training overview
    overview = Table(title="Training Overview", box=box.ROUNDED)
    overview.add_column("Property", style="cyan")
    overview.add_column("Value", style="green")
    overview.add_row("Model", config.get("model", "N/A"))
    overview.add_row("Dataset", f"{dataset_info['project']} v{dataset_info['version']}")
    overview.add_row("Task", config.get("task", "detect"))
    overview.add_row("Epochs", str(config.get("epochs", "N/A")))
    overview.add_row("Batch Size", str(config.get("batch", "N/A")))
    overview.add_row("Image Size", str(config.get("imgsz", "N/A")))
    overview.add_row("Device", str(config.get("device", "N/A")))
    overview.add_row("Best Model", train_info["model_path"])

    # Training time (Phase 3.3)
    elapsed = train_info.get("elapsed_seconds", 0)
    if elapsed > 0:
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        if hours > 0:
            overview.add_row("Training Time", f"{hours}h {minutes}m")
        else:
            overview.add_row("Training Time", f"{minutes}m {int(elapsed % 60)}s")

    # Model file size
    best_pt = Path(train_info["model_path"])
    if best_pt.exists():
        overview.add_row("Model Size", _format_size(best_pt.stat().st_size))

    console.print(overview)
    console.print()

    # Metrics from CSV
    results_csv = train_dir / "results.csv"
    if results_csv.exists():
        _display_metrics_from_csv(results_csv)
    else:
        # Fallback: extract metrics directly from YOLO results object
        results_obj = train_info.get("results")
        if results_obj and hasattr(results_obj, "results_dict"):
            metrics_table = Table(title="Final Training Metrics", box=box.ROUNDED)
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green")
            for key, value in results_obj.results_dict.items():
                try:
                    metrics_table.add_row(key, f"{float(value):.5f}")
                except (ValueError, TypeError):
                    metrics_table.add_row(key, str(value))
            console.print(metrics_table)
            console.print()
        else:
            console.print("[yellow]No training metrics found (results.csv missing).[/]")
            console.print()

    # Output files
    if not train_dir.exists():
        console.print("[yellow]Training directory not found — cannot list output files.[/]")
        console.print(f"[dim]Expected: {train_dir}[/]")
        console.print()
        return

    files_table = Table(title="Output Files", box=box.ROUNDED)
    files_table.add_column("File", style="cyan")
    files_table.add_column("Size", style="green")

    file_count = 0
    for item in sorted(train_dir.rglob("*")):
        if item.is_file():
            size_str = _format_size(item.stat().st_size)
            files_table.add_row(str(item.relative_to(train_dir)), size_str)
            file_count += 1

    if file_count == 0:
        console.print("[yellow]No output files found in training directory.[/]")
        console.print(f"[dim]Directory: {train_dir}[/]")
    else:
        console.print(files_table)
    console.print()


# ─── Step 6: Export Model (Phase 3.2) ────────────────────────────────────────


def step_export_model(train_info: dict, config: dict) -> list[str]:
    console.print(Rule("[bold cyan]Step 6: Export Model[/]"))
    console.print()

    do_export = Confirm.ask("[yellow]Export model to deployment formats?[/]", default=True)
    if not do_export:
        return []

    console.print()
    console.print("[bold]Available export formats:[/]")
    format_keys = list(EXPORT_FORMATS.keys())
    for i, key in enumerate(format_keys, 1):
        console.print(f"  [dim]{i}.[/] [bold]{key}[/] — {EXPORT_FORMATS[key]}")

    console.print()
    console.print("[dim]Enter numbers separated by commas (e.g., 1,2,4)[/]")
    selection = Prompt.ask("[yellow]Select formats[/]", default="1")

    selected_indices = [int(s.strip()) for s in selection.split(",") if s.strip().isdigit()]
    selected_formats = [format_keys[i - 1] for i in selected_indices if 1 <= i <= len(format_keys)]

    if not selected_formats:
        console.print("[yellow]No valid formats selected. Skipping export.[/]")
        return []

    # Export options
    console.print()
    half = Confirm.ask("[yellow]Export with FP16 (half precision)?[/]", default=True)
    dynamic = Confirm.ask("[yellow]Enable dynamic input shapes?[/]", default=False)
    int8 = False

    if "engine" in selected_formats or "tflite" in selected_formats:
        int8 = Confirm.ask("[yellow]Enable INT8 quantization?[/]", default=False)

    console.print()

    from ultralytics import YOLO

    model = YOLO(train_info["model_path"])
    exported_paths = []

    for fmt in selected_formats:
        console.print(f"[bold]Exporting to {fmt}...[/]")
        try:
            export_args = {
                "format": fmt,
                "half": half,
                "dynamic": dynamic,
                "imgsz": config.get("imgsz", 640),
            }
            if int8:
                export_args["int8"] = True
                if config.get("data"):
                    export_args["data"] = config["data"]

            path = model.export(**export_args)
            exported_paths.append(str(path))
            console.print(f"  [green]Exported: {path}[/]")
        except Exception as e:
            console.print(f"  [red]Failed to export {fmt}: {e}[/]")

    console.print()

    if exported_paths:
        export_table = Table(title="Exported Models", box=box.ROUNDED)
        export_table.add_column("Format", style="cyan")
        export_table.add_column("Path", style="green")
        export_table.add_column("Size", style="yellow")

        for path in exported_paths:
            p = Path(path)
            fmt = p.suffix.lstrip(".")
            size = _format_size(p.stat().st_size) if p.exists() else "N/A"
            export_table.add_row(fmt, str(p.name), size)

        console.print(export_table)
        console.print()

    return exported_paths


# ─── Step 7: Zip Results ──────────────────────────────────────────────────────


def step_zip_results(train_info: dict, config: dict) -> str:
    console.print(Rule("[bold cyan]Step 7: Export Results (Zip)[/]"))
    console.print()

    do_zip = Confirm.ask("[yellow]Create zip archive of results?[/]", default=True)
    if not do_zip:
        return ""

    train_dir = Path(train_info["train_dir"])

    if not train_dir.exists():
        console.print(f"[bold red]Error:[/] Training directory not found: {train_dir}")
        console.print("[yellow]Cannot create zip archive without training output files.[/]")
        return ""

    zip_name = f"{config.get('name', 'results')}_results.zip"
    zip_path = train_dir.parent / zip_name
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    include_weights = Confirm.ask(
        "[yellow]Include model weights (best.pt & last.pt)?[/]", default=True
    )
    include_plots = Confirm.ask(
        "[yellow]Include plots and visualizations?[/]", default=True
    )
    include_all = Confirm.ask(
        "[yellow]Include all other files (csv, args, etc.)?[/]", default=True
    )

    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        zip_task = progress.add_task("Creating zip archive...", total=None)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in sorted(train_dir.rglob("*")):
                if not item.is_file():
                    continue

                rel_str = str(item.relative_to(train_dir))

                if not include_weights and "weights" in rel_str:
                    continue
                if not include_plots and rel_str.endswith((".png", ".jpg")):
                    continue
                if not include_all and not any(
                    rel_str.endswith(ext)
                    for ext in [".pt", ".png", ".jpg", ".csv", ".yaml"]
                ):
                    continue

                zf.write(item, arcname=rel_str)

        progress.update(zip_task, description="[green]Zip created!")

    zip_size = _format_size(zip_path.stat().st_size)

    console.print()
    console.print(
        Panel(
            f"[bold green]Exported:[/] {zip_path}\n[bold green]Size:[/] {zip_size}",
            title="Export Complete",
            border_style="green",
        )
    )
    console.print()

    return str(zip_path)


# ─── Hyperparameter Tuning Mode (Phase 4.3) ──────────────────────────────────


def mode_tune(dataset_info: dict) -> None:
    """Automated hyperparameter search using model.tune()."""
    console.print(Rule("[bold magenta]Hyperparameter Tuning Mode[/]"))
    console.print()
    console.print("[dim]This mode uses Ultralytics' built-in hyperparameter evolution[/]")
    console.print("[dim]to find optimal training parameters for your dataset.[/]")
    console.print()

    model_name = _select_model()

    from ultralytics import YOLO

    model = YOLO(model_name)

    iterations = IntPrompt.ask("[yellow]Tuning iterations[/]", default=300)
    tune_epochs = IntPrompt.ask("[yellow]Epochs per iteration[/]", default=30)
    imgsz = IntPrompt.ask("[yellow]Image size[/]", default=640)

    detected_device, device_desc = _detect_device()
    console.print(f"[bold]Device:[/] [green]{device_desc}[/]")
    console.print()

    console.print("[bold]Starting hyperparameter tuning...[/]")
    console.print(f"[dim]This will run {iterations} iterations x {tune_epochs} epochs each.[/]")
    console.print(f"[dim]Estimated time: {iterations * tune_epochs * 0.5:.0f}+ minutes (varies by hardware)[/]")
    console.print()

    if not Confirm.ask("[yellow]Proceed?[/]", default=True):
        return

    results = model.tune(
        data=dataset_info["data_yaml"],
        epochs=tune_epochs,
        iterations=iterations,
        imgsz=imgsz,
        device=detected_device,
        plots=True,
        val=True,
    )

    console.print()
    console.print("[bold green]Tuning complete![/]")
    console.print("[dim]Best hyperparameters saved to runs/detect/tune/best_hyperparameters.yaml[/]")
    console.print()


# ─── Load Config from YAML (Phase 4.1) ───────────────────────────────────────


def _load_training_config() -> dict | None:
    """Check for existing YAML configs and offer to load one."""
    yaml_files = list(Path(".").glob("config-*.yaml")) + list(Path(".").glob("training_config*.yaml"))

    if not yaml_files:
        return None

    console.print("[bold]Found saved configurations:[/]")
    for i, f in enumerate(yaml_files, 1):
        console.print(f"  [dim]{i}.[/] {f.name}")
    console.print()

    choice = Prompt.ask(
        "[yellow]Load a saved config?[/]",
        choices=["skip"] + [str(i) for i in range(1, len(yaml_files) + 1)],
        default="skip",
    )

    if choice == "skip":
        return None

    selected_file = yaml_files[int(choice) - 1]
    try:
        with open(selected_file) as f:
            config = yaml.safe_load(f)
        console.print(f"[green]Loaded configuration from {selected_file}[/]")
        return config
    except Exception as e:
        console.print(f"[red]Failed to load config: {e}[/]")
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                "[bold white]YOLO Training Wizard[/]\n"
                "[dim]Train YOLO models with Roboflow datasets — optimized for best practices[/]\n\n"
                "[cyan]1.[/] Download Dataset from Roboflow\n"
                "[cyan]2.[/] Configure Training (with presets)\n"
                "[cyan]3.[/] Train Model\n"
                "[cyan]4.[/] Validate Model\n"
                "[cyan]5.[/] View Training Summary\n"
                "[cyan]6.[/] Export Model (ONNX/TensorRT/etc.)\n"
                "[cyan]7.[/] Export Results (Zip)\n\n"
                "[dim]Features: Auto-batch, Resume, YOLO26 defaults, Augmentation presets,[/]\n"
                "[dim]Health checks, Model export, HP tuning mode[/]"
            ),
            border_style="bright_blue",
            padding=(1, 2),
        )
    )
    console.print()

    # Check for --tune flag
    if "--tune" in sys.argv:
        console.print("[bold magenta]Entering Hyperparameter Tuning Mode[/]")
        console.print()
        # Still need dataset
        dataset_info = step_download_dataset()
        mode_tune(dataset_info)
        return

    try:
        resume_info = step_check_resume()
        exported = []
        zip_path = ""

        if resume_info:
            from ultralytics import YOLO

            train_info = step_train({}, resume_info=resume_info)

            args_yaml = Path(resume_info["dir"]) / "args.yaml"
            config = {}
            if args_yaml.exists():
                with open(args_yaml) as f:
                    config = yaml.safe_load(f) or {}

            dataset_info = {
                "project": config.get("name", "resumed"),
                "version": "N/A",
                "classes": [],
                "data_yaml": config.get("data", ""),
            }

            step_validate(train_info, config)
            step_summary(train_info, config, dataset_info)
            exported = step_export_model(train_info, config)
            zip_path = step_zip_results(train_info, config)

        else:
            saved_config = _load_training_config()
            dataset_info = step_download_dataset()

            if saved_config:
                config = saved_config
                config["data"] = dataset_info["data_yaml"]
                console.print()

                table = Table(title="Loaded Configuration", box=box.ROUNDED)
                table.add_column("Parameter", style="cyan")
                table.add_column("Value", style="green")
                for key, value in sorted(config.items()):
                    table.add_row(key, str(value))
                console.print(table)
                console.print()

                if not Confirm.ask("[yellow]Use this configuration?[/]", default=True):
                    config = step_configure_training(dataset_info)
            else:
                config = step_configure_training(dataset_info)

            step_health_checks(config, dataset_info)
            train_info = step_train(config)
            step_validate(train_info, config)
            step_summary(train_info, config, dataset_info)
            exported = step_export_model(train_info, config)
            zip_path = step_zip_results(train_info, config)

        summary_lines = [
            "[bold green]All done! Your YOLO model has been trained and exported.[/]\n",
            f"[cyan]Best model:[/] {train_info['model_path']}",
        ]
        if exported:
            summary_lines.append(f"[cyan]Exported formats:[/] {len(exported)}")
        if zip_path:
            summary_lines.append(f"[cyan]Zip export:[/] {zip_path}")

        console.print(
            Panel(
                "\n".join(summary_lines),
                title="Wizard Complete",
                border_style="bright_green",
                padding=(1, 2),
            )
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Wizard cancelled by user.[/]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/] {e}")
        if "--verbose" in sys.argv:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
