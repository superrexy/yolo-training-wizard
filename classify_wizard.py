import os
import re
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

CONFIG_DIR = Path.home() / ".yolo-wizard"
CONFIG_FILE = CONFIG_DIR / "config.json"

CLS_MODELS = [
    "yolo26n-cls", "yolo26s-cls", "yolo26m-cls", "yolo26l-cls", "yolo26x-cls",
    "yolo11n-cls", "yolo11s-cls", "yolo11m-cls", "yolo11l-cls", "yolo11x-cls",
    "yolov8n-cls", "yolov8s-cls", "yolov8m-cls", "yolov8l-cls", "yolov8x-cls",
]

OUTPUT_DIR = Path("runs")
DATASETS_DIR = Path("datasets")

CLS_TRAINING_PRESETS = {
    "quick": {"label": "Quick Test", "description": "Fast validation that everything works (10 epochs)", "epochs": 10, "imgsz": 224, "batch": 64, "patience": 5},
    "balanced": {"label": "Balanced", "description": "Good quality with reasonable training time", "epochs": 50, "imgsz": 224, "batch": 32, "patience": 15},
    "max_quality": {"label": "Maximum Quality", "description": "Best possible model \u2014 long training time", "epochs": 150, "imgsz": 224, "batch": 16, "patience": 30, "cos_lr": True},
    "finetune": {"label": "Fine-Tune", "description": "Fine-tune a pretrained model on your data (frozen backbone)", "epochs": 30, "imgsz": 224, "batch": 32, "patience": 10, "freeze": 10},
    "custom": {"label": "Custom", "description": "Configure everything manually"},
}

CLS_AUGMENTATION_PRESETS = {
    "none": {"label": "None", "description": "No augmentation (for debugging or very clean datasets)", "hsv_h": 0, "hsv_s": 0, "hsv_v": 0, "degrees": 0, "translate": 0, "scale": 0, "flipud": 0, "fliplr": 0, "erasing": 0},
    "light": {"label": "Light", "description": "Gentle augmentation for small/clean datasets", "hsv_h": 0.01, "hsv_s": 0.3, "hsv_v": 0.2, "degrees": 5, "translate": 0.05, "scale": 0.1, "flipud": 0, "fliplr": 0.5, "erasing": 0.1},
    "medium": {"label": "Medium (Default)", "description": "Balanced augmentation \u2014 good for most cases", "hsv_h": 0.015, "hsv_s": 0.5, "hsv_v": 0.3, "degrees": 10, "translate": 0.1, "scale": 0.3, "flipud": 0.1, "fliplr": 0.5, "erasing": 0.2},
    "heavy": {"label": "Heavy", "description": "Aggressive augmentation for large models or overfitting prevention", "hsv_h": 0.02, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 15, "translate": 0.15, "scale": 0.5, "flipud": 0.3, "fliplr": 0.5, "erasing": 0.4},
    "custom": {"label": "Custom", "description": "Configure each augmentation parameter individually"},
}

YOLO26_DEFAULTS = {
    "n": {"lr0": 0.0054, "lrf": 0.0495, "weight_decay": 0.00064, "momentum": 0.947},
    "s": {"lr0": 0.00038, "lrf": 0.882, "weight_decay": 0.00027, "momentum": 0.948},
    "m": {"lr0": 0.00038, "lrf": 0.882, "weight_decay": 0.00027, "momentum": 0.948},
    "l": {"lr0": 0.00038, "lrf": 0.882, "weight_decay": 0.00027, "momentum": 0.948},
    "x": {"lr0": 0.00038, "lrf": 0.882, "weight_decay": 0.00027, "momentum": 0.948},
}

EXPORT_FORMATS = {
    "onnx": {"suffix": ".onnx", "desc": "ONNX (cross-platform)"},
    "engine": {"suffix": ".engine", "desc": "TensorRT (NVIDIA GPU)"},
    "coreml": {"suffix": ".mlpackage", "desc": "CoreML (Apple)"},
    "tflite": {"suffix": ".tflite", "desc": "TFLite (Mobile/Edge)"},
    "openvino": {"suffix": "_openvino_model", "desc": "OpenVINO (Intel)"},
    "ncnn": {"suffix": "_ncnn_model", "desc": "NCNN (Mobile)"},
    "torchscript": {"suffix": ".torchscript", "desc": "TorchScript"},
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


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
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


def _prompt_api_key() -> str:
    saved = _load_config()
    saved_key = saved.get("roboflow_api_key")

    if saved_key:
        masked = _mask_key(saved_key)
        console.print(f"[dim]Saved API key found:[/] [bold]{masked}[/]")
        console.print()
        choice = Prompt.ask("[yellow]Use saved API key?[/]", choices=["yes", "new"], default="yes")
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


def _normalize_split_dirs(dataset_path: Path) -> None:
    valid_dir = dataset_path / "valid"
    val_dir = dataset_path / "val"
    if valid_dir.is_dir() and not val_dir.exists():
        valid_dir.rename(val_dir)


def _validate_cls_dataset(dataset_path: Path) -> dict:
    """Validate classification dataset structure (folder-based).
    Returns dict with 'classes' list, 'splits' dict of {split: {class: count}}.
    """
    if not dataset_path.exists():
        console.print(f"[bold red]Error:[/] Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    if not dataset_path.is_dir():
        console.print(f"[bold red]Error:[/] Dataset path is not a directory: {dataset_path}")
        sys.exit(1)
    _normalize_split_dirs(dataset_path)

    possible_splits = ["train", "val", "test"]
    found_splits: dict[str, Path] = {}
    for split_name in possible_splits:
        split_path = dataset_path / split_name
        if split_path.is_dir():
            found_splits[split_name] = split_path

    if "train" not in found_splits:
        console.print("[bold red]Error:[/] 'train' split directory not found.")
        console.print(f"[dim]Expected: {dataset_path / 'train'}[/]")
        console.print("[dim]Classification datasets must have at least a train/ folder with class subfolders.[/]")
        sys.exit(1)

    train_path = found_splits["train"]
    classes = sorted([d.name for d in train_path.iterdir() if d.is_dir() and not d.name.startswith(".")])

    if len(classes) < 2:
        console.print(f"[bold red]Error:[/] Found only {len(classes)} class(es) in train/.")
        console.print("[dim]Classification requires at least 2 classes.[/]")
        sys.exit(1)

    splits_stats: dict[str, dict[str, int]] = {}
    for split_name, split_path in found_splits.items():
        class_counts: dict[str, int] = {}
        for class_name in classes:
            class_dir = split_path / class_name
            if class_dir.is_dir():
                count = sum(1 for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
                class_counts[class_name] = count
            else:
                class_counts[class_name] = 0
        splits_stats[split_name] = class_counts

    for split_name in ["val", "test"]:
        if split_name in splits_stats:
            missing = [c for c in classes if splits_stats[split_name].get(c, 0) == 0]
            if missing:
                console.print(f"[yellow]Warning:[/] {len(missing)} class(es) have no images in {split_name}/: " + ", ".join(missing[:5]) + ("..." if len(missing) > 5 else ""))

    return {"classes": classes, "splits": splits_stats, "path": str(dataset_path)}


def _get_cls_dataset_stats(dataset_path: Path) -> dict[str, dict[str, int]]:
    """Count images per class per split."""
    stats: dict[str, dict[str, int]] = {}
    for split_name in ["train", "val", "test"]:
        split_path = dataset_path / split_name
        if not split_path.is_dir():
            continue
        class_counts: dict[str, int] = {}
        for class_dir in sorted(split_path.iterdir()):
            if class_dir.is_dir() and not class_dir.name.startswith("."):
                count = sum(1 for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
                class_counts[class_dir.name] = count
        stats[split_name] = class_counts
    return stats


def _format_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _detect_device() -> tuple[str, str]:
    """Auto-detect best available training device."""
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
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
    usage = shutil.disk_usage(".")
    return usage.free / (1024**3)


def _check_resumable_runs() -> list[dict]:
    """Scan runs/classify/ for interrupted training sessions."""
    resumable = []
    runs_dir = OUTPUT_DIR / "classify"
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
                completed_epochs = sum(1 for _ in csv.reader(f)) - 1
            with open(args_yaml) as f:
                args = yaml.safe_load(f)
            total_epochs = args.get("epochs", 0)
            if completed_epochs > 0 and completed_epochs < total_epochs:
                resumable.append({"path": str(last_pt), "dir": str(train_dir), "model": args.get("model", "unknown"), "completed": completed_epochs, "total": total_epochs, "data": args.get("data", "")})
        except Exception:
            continue
    return resumable


def _get_model_size_key(model_name: str) -> str:
    clean = model_name.replace(".pt", "").replace("-cls", "")
    for size in ["n", "s", "m", "l", "x"]:
        if clean.endswith(size):
            return size
    return "s"


def _is_yolo26(model_name: str) -> bool:
    return "yolo26" in model_name.lower()


def _get_gpu_count(device: str) -> int:
    if "," in device:
        return len([d.strip() for d in device.split(",") if d.strip()])
    return 1 if device not in ("cpu", "mps") else 0


def _is_multi_gpu(device: str) -> bool:
    return "," in device


def _parse_batch_input(batch_str: str, device: str) -> int | float:
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
        return int(batch_str)
    except ValueError:
        return -1


def _run_cls_health_checks(config: dict, dataset_info: dict) -> list[tuple[str, str]]:
    checks = []
    free_gb = _check_disk_space()
    if free_gb < 5:
        checks.append(("error", f"Very low disk space: {free_gb:.1f}GB free. Training may fail."))
    elif free_gb < 15:
        checks.append(("warning", f"Low disk space: {free_gb:.1f}GB free. Consider freeing space."))
    else:
        checks.append(("info", f"Disk space: {free_gb:.1f}GB free"))

    splits = dataset_info.get("splits", {})
    train_stats = splits.get("train", {})
    val_stats = splits.get("val", {})
    train_total = sum(train_stats.values())
    val_total = sum(val_stats.values())

    if train_total > 0:
        checks.append(("info", f"Training images: {train_total}"))
    else:
        checks.append(("error", "No training images found."))
    if val_total > 0:
        checks.append(("info", f"Validation images: {val_total}"))
    else:
        checks.append(("warning", "No validation split found. Model will use auto-split from training data."))

    if 0 < train_total < 100:
        checks.append(("warning", f"Very small dataset ({train_total} images). Consider using 'finetune' preset with frozen backbone."))
    elif 0 < train_total < 500:
        checks.append(("info", "Small dataset. Heavy augmentation recommended."))

    classes = dataset_info.get("classes", [])
    num_classes = len(classes)
    if num_classes > 0:
        checks.append(("info", f"Classes: {num_classes}"))
        if num_classes > 100:
            checks.append(("warning", f"Many classes ({num_classes}). Consider using a larger model (m/l/x)."))

    if train_stats:
        counts = [v for v in train_stats.values() if v > 0]
        if counts:
            max_count = max(counts)
            min_count = min(counts)
            if min_count > 0 and max_count / min_count > 10:
                checks.append(("warning", f"Class imbalance detected: ratio {max_count}:{min_count} (>10:1). Consider oversampling minority classes or using class weights."))

    batch = config.get("batch", 32)
    device = config.get("device", "cpu")
    if batch == -1 and device == "cpu":
        checks.append(("warning", "Auto-batch requires GPU. Falling back to batch=16 for CPU."))
    if (batch == -1 or isinstance(batch, float)) and _is_multi_gpu(device):
        gpu_count = _get_gpu_count(device)
        checks.append(("error", f"Auto-batch not supported for Multi-GPU training. Specify batch size as multiple of {gpu_count}."))

    imgsz = config.get("imgsz", 224)
    if imgsz != 224:
        checks.append(("warning", f"Image size is {imgsz} (default for classification is 224). Non-standard size may affect pretrained weight compatibility."))
    else:
        checks.append(("info", f"Image size: {imgsz} (standard for classification)"))

    epochs = config.get("epochs", 50)
    patience = config.get("patience", 15)
    if patience >= epochs:
        checks.append(("warning", f"Patience ({patience}) >= epochs ({epochs}). Early stopping will never trigger."))

    return checks


def _parse_roboflow_snippet(snippet: str) -> dict | None:
    result = {}
    api_key_match = re.search(r'Roboflow\(\s*api_key\s*=\s*["\']([^"\']+)["\']', snippet)
    if api_key_match:
        result["api_key"] = api_key_match.group(1)
    wp_match = re.search(r'\.workspace\(\s*["\']([^"\']+)["\']\s*\)\s*\.\s*project\(\s*["\']([^"\']+)["\']\s*\)', snippet)
    if wp_match:
        result["workspace"] = wp_match.group(1)
        result["project"] = wp_match.group(2)
    version_match = re.search(r'\.version\(\s*(\d+)\s*\)', snippet)
    if version_match:
        result["version"] = int(version_match.group(1))
    format_match = re.search(r'\.download\(\s*["\']([^"\']+)["\']\s*\)', snippet)
    if format_match:
        result["format"] = format_match.group(1)
    if "api_key" in result and "workspace" in result and "project" in result:
        result.setdefault("version", 1)
        result.setdefault("format", "folder")
        return result
    return None


def _prompt_roboflow_snippet() -> dict | None:
    console.print(Panel("[bold]Paste your Roboflow snippet below[/]\n[dim]Copy the code from Roboflow's 'Download Dataset' page.\nPaste all lines, then press Enter on an empty line to finish.\nType 'skip' to enter details manually instead.[/]", border_style="yellow", padding=(0, 1)))
    console.print()

    lines: list[str] = []
    try:
        while True:
            line = input()
            if line.strip().lower() == "skip":
                return None
            if not line.strip() and lines:
                snippet = "\n".join(lines)
                parsed = _parse_roboflow_snippet(snippet)
                if parsed:
                    break
                lines.append(line)
                continue
            lines.append(line)
    except EOFError:
        pass

    if not lines:
        return None

    snippet = "\n".join(lines)
    parsed = _parse_roboflow_snippet(snippet)

    if parsed:
        console.print()
        console.print("[green]Snippet parsed successfully![/]")
        table = Table(title="Parsed from Snippet", box=box.ROUNDED)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("API Key", _mask_key(parsed["api_key"]))
        table.add_row("Workspace", parsed["workspace"])
        table.add_row("Project", parsed["project"])
        table.add_row("Version", str(parsed["version"]))
        table.add_row("Format", parsed["format"])
        console.print(table)
        console.print()
        if Confirm.ask("[yellow]Use these settings?[/]", default=True):
            return parsed
        else:
            console.print("[dim]Switching to manual input...[/]")
            return None
    else:
        console.print("[red]Could not parse the snippet. Switching to manual input...[/]")
        console.print("[dim]Make sure you paste the full Roboflow code block including the api_key line.[/]")
        console.print()
        return None


def _is_detection_dataset(dataset_path: Path) -> bool:
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        return False
    for split_name in ["train", "valid", "test"]:
        labels_dir = dataset_path / split_name / "labels"
        if labels_dir.is_dir():
            return True
    return False


def _is_cls_dataset(dataset_path: Path) -> bool:
    train_dir = dataset_path / "train"
    if not train_dir.is_dir():
        return False
    subdirs = [d for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not subdirs:
        return False
    subdir_names = {d.name for d in subdirs}
    if subdir_names == {"images", "labels"} or "labels" in subdir_names:
        return False
    for subdir in subdirs:
        has_images = any(f.suffix.lower() in IMAGE_EXTENSIONS for f in subdir.iterdir() if f.is_file())
        if has_images:
            return True
    return False


def _convert_detection_to_cls(detection_path: Path, output_path: Path) -> Path:
    converter_script = Path(__file__).parent / "utils" / "convert_to_classification.py"
    if not converter_script.exists():
        console.print(f"[bold red]Error:[/] utils/convert_to_classification.py not found at {converter_script}")
        sys.exit(1)

    console.print()
    console.print("[bold]Converting detection dataset to classification format...[/]")
    console.print(f"[dim]Input:  {detection_path}[/]")
    console.print(f"[dim]Output: {output_path}[/]")
    console.print()

    cmd = [sys.executable, str(converter_script), "--input", str(detection_path), "--output", str(output_path), "--overwrite"]
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=600)
        if result.returncode != 0:
            console.print("[bold red]Error:[/] Conversion failed.")
            sys.exit(1)
        console.print()
        console.print("[bold green]Conversion complete![/]")
        return output_path
    except subprocess.TimeoutExpired:
        console.print("[bold red]Error:[/] Conversion timed out after 10 minutes.")
        sys.exit(1)
    except OSError as e:
        console.print(f"[bold red]Error:[/] Failed to run converter: {e}")
        sys.exit(1)


def step_check_resume() -> dict | None:
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

    resume_choice = Prompt.ask("[yellow]Resume a previous run?[/]", choices=["skip"] + [str(i) for i in range(1, len(resumable) + 1)], default="skip")
    if resume_choice == "skip":
        return None

    selected = resumable[int(resume_choice) - 1]
    console.print(f"[green]Resuming training from {selected['path']}[/]")
    console.print()
    return selected


def step_download_dataset() -> dict:
    console.print(Rule("[bold cyan]Step 1: Download/Select Classification Dataset[/]"))
    console.print()

    input_method = Prompt.ask("[yellow]How would you like to provide the dataset?[/]", choices=["paste", "manual", "path"], default="paste")
    dataset_path = None

    if input_method == "path":
        console.print()
        path_str = Prompt.ask("[yellow]Enter path to classification dataset[/]")
        dataset_path = Path(path_str).resolve()
        if not dataset_path.exists():
            console.print(f"[bold red]Error:[/] Path does not exist: {dataset_path}")
            sys.exit(1)
        if _is_cls_dataset(dataset_path):
            console.print("[green]Dataset is already in classification format.[/]")
        elif _is_detection_dataset(dataset_path):
            console.print("[yellow]Detection format detected. Converting to classification...[/]")
            cls_path = dataset_path.parent / (dataset_path.name + "-cls")
            dataset_path = _convert_detection_to_cls(dataset_path, cls_path)
        else:
            console.print("[bold red]Error:[/] Unrecognized dataset format.")
            console.print("[dim]Expected classification format: train/class_name/*.jpg[/]")
            console.print("[dim]Or detection format with data.yaml for auto-conversion.[/]")
            sys.exit(1)

        dataset_info = _validate_cls_dataset(dataset_path)
        project_name = dataset_path.name
        info = {"project": project_name, "version": "local", "path": str(dataset_path), "classes": dataset_info["classes"], "splits": dataset_info["splits"]}
    else:
        snippet_data = None
        if input_method == "paste":
            console.print()
            snippet_data = _prompt_roboflow_snippet()

        if snippet_data:
            api_key = snippet_data["api_key"]
            workspace = snippet_data["workspace"]
            project_name = snippet_data["project"]
            version_number = snippet_data["version"]
            dataset_format = snippet_data["format"]
            save_key = Confirm.ask("[yellow]Save this API key for future use?[/]", default=True)
            if save_key:
                config = _load_config()
                config["roboflow_api_key"] = api_key
                _save_config(config)
                console.print(f"[green]API key saved to {CONFIG_FILE}[/]")
        else:
            api_key = _prompt_api_key()
            console.print()
            workspace = Prompt.ask("[yellow]Workspace name[/]")
            project_name = Prompt.ask("[yellow]Project name[/]")
            version_number = IntPrompt.ask("[yellow]Dataset version[/]", default=1)
            dataset_format = Prompt.ask("[yellow]Export format[/]", default="folder", choices=["folder", "yolo26", "yolov12", "yolov11", "yolov8", "yolov5", "yolov7", "yolov9", "coco", "voc"])

        console.print()
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
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

        downloaded_path = Path(dataset_location)
        if _is_cls_dataset(downloaded_path):
            console.print("[green]Dataset is already in classification format.[/]")
            dataset_path = downloaded_path
        elif _is_detection_dataset(downloaded_path):
            console.print()
            console.print("[yellow]Detection format detected. Converting to classification format...[/]")
            cls_path = downloaded_path.parent / (downloaded_path.name + "-cls")
            dataset_path = _convert_detection_to_cls(downloaded_path, cls_path)
        else:
            console.print("[bold red]Error:[/] Downloaded dataset is not in a recognized format.")
            console.print("[dim]Expected detection format (data.yaml + labels/) or classification format (train/class_name/).[/]")
            sys.exit(1)

        dataset_info = _validate_cls_dataset(dataset_path)
        info = {"workspace": workspace, "project": project_name, "version": version_number, "format": dataset_format, "path": str(dataset_path), "classes": dataset_info["classes"], "splits": dataset_info["splits"]}

    console.print()
    table = Table(title="Classification Dataset Info", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Project", info.get("project", "N/A"))
    table.add_row("Version", str(info.get("version", "N/A")))
    table.add_row("Location", info["path"])
    table.add_row("Classes", ", ".join(info["classes"][:20]) + ("..." if len(info["classes"]) > 20 else ""))
    table.add_row("Num Classes", str(len(info["classes"])))

    splits = info.get("splits", {})
    for split_name in ["train", "val", "test"]:
        if split_name in splits:
            total = sum(splits[split_name].values())
            table.add_row(f"{split_name.capitalize()} Images", str(total))
    console.print(table)

    if splits.get("train"):
        console.print()
        class_table = Table(title="Images per Class", box=box.ROUNDED)
        class_table.add_column("Class", style="cyan")
        class_table.add_column("Train", style="green", justify="right")
        if "val" in splits:
            class_table.add_column("Val", style="yellow", justify="right")
        if "test" in splits:
            class_table.add_column("Test", style="dim", justify="right")
        for cls_name in info["classes"]:
            row = [cls_name, str(splits["train"].get(cls_name, 0))]
            if "val" in splits:
                row.append(str(splits["val"].get(cls_name, 0)))
            if "test" in splits:
                row.append(str(splits["test"].get(cls_name, 0)))
            class_table.add_row(*row)
        total_row = ["[bold]Total[/]", f"[bold]{sum(splits['train'].values())}[/]"]
        if "val" in splits:
            total_row.append(f"[bold]{sum(splits['val'].values())}[/]")
        if "test" in splits:
            total_row.append(f"[bold]{sum(splits['test'].values())}[/]")
        class_table.add_row(*total_row)
        console.print(class_table)

    console.print()
    return info


def _select_cls_model() -> str:
    console.print("[bold]Available classification models:[/]")
    console.print()
    groups = {"YOLO26 (Latest)": [m for m in CLS_MODELS if "yolo26" in m], "YOLO11": [m for m in CLS_MODELS if "yolo11" in m], "YOLOv8": [m for m in CLS_MODELS if "yolov8" in m]}
    idx = 1
    model_map: dict[int, str] = {}
    for group_name, models in groups.items():
        console.print(f"  [bold dim]{group_name}:[/]")
        for model in models:
            size = _get_model_size_key(model)
            size_desc = {"n": "nano", "s": "small", "m": "medium", "l": "large", "x": "xlarge"}
            console.print(f"    [dim]{idx:2d}.[/] {model} [dim]({size_desc.get(size, size)})[/]")
            model_map[idx] = model
            idx += 1
        console.print()
    model_idx = IntPrompt.ask("[yellow]Select model number[/]", default=1, choices=[str(i) for i in range(1, idx)])
    return model_map[model_idx]


def _select_cls_preset() -> dict:
    console.print("[bold]Training Presets:[/]")
    console.print()
    preset_keys = list(CLS_TRAINING_PRESETS.keys())
    for i, key in enumerate(preset_keys, 1):
        preset = CLS_TRAINING_PRESETS[key]
        console.print(f"  [dim]{i}.[/] [bold]{preset['label']}[/] \u2014 [dim]{preset['description']}[/]")
    console.print()
    preset_idx = IntPrompt.ask("[yellow]Select preset[/]", default=2, choices=[str(i) for i in range(1, len(preset_keys) + 1)])
    selected_key = preset_keys[preset_idx - 1]
    selected = CLS_TRAINING_PRESETS[selected_key].copy()
    selected.pop("label", None)
    selected.pop("description", None)
    if selected_key != "custom":
        console.print(f"[green]Using preset: {CLS_TRAINING_PRESETS[selected_key]['label']}[/]")
    return {"_preset_key": selected_key, **selected}


def _configure_cls_augmentation() -> dict:
    console.print()
    console.print("[bold]Augmentation Level:[/]")
    console.print()
    aug_keys = list(CLS_AUGMENTATION_PRESETS.keys())
    for i, key in enumerate(aug_keys, 1):
        preset = CLS_AUGMENTATION_PRESETS[key]
        console.print(f"  [dim]{i}.[/] [bold]{preset['label']}[/] \u2014 [dim]{preset['description']}[/]")
    console.print()
    aug_idx = IntPrompt.ask("[yellow]Select augmentation level[/]", default=3, choices=[str(i) for i in range(1, len(aug_keys) + 1)])
    selected_key = aug_keys[aug_idx - 1]
    selected = CLS_AUGMENTATION_PRESETS[selected_key].copy()
    selected.pop("label", None)
    selected.pop("description", None)
    if selected_key == "custom":
        selected = {"hsv_h": FloatPrompt.ask("[yellow]  HSV-Hue[/]", default=0.015), "hsv_s": FloatPrompt.ask("[yellow]  HSV-Saturation[/]", default=0.5), "hsv_v": FloatPrompt.ask("[yellow]  HSV-Value[/]", default=0.3), "degrees": FloatPrompt.ask("[yellow]  Rotation degrees[/]", default=10.0), "translate": FloatPrompt.ask("[yellow]  Translation[/]", default=0.1), "scale": FloatPrompt.ask("[yellow]  Scale[/]", default=0.3), "flipud": FloatPrompt.ask("[yellow]  Flip up-down prob[/]", default=0.0), "fliplr": FloatPrompt.ask("[yellow]  Flip left-right prob[/]", default=0.5), "erasing": FloatPrompt.ask("[yellow]  Random erasing prob[/]", default=0.2)}
    return selected


def _configure_cls_optimizer_lr(model_name: str) -> dict:
    console.print()
    console.print("[bold]Optimizer & Learning Rate:[/]")
    console.print()
    is_26 = _is_yolo26(model_name)
    size_key = _get_model_size_key(model_name)
    if is_26 and size_key in YOLO26_DEFAULTS:
        defaults = YOLO26_DEFAULTS[size_key]
        console.print(f"[dim]  Using YOLO26-{size_key} optimized defaults[/]")
    else:
        defaults = {"lr0": 0.01, "lrf": 0.01, "weight_decay": 0.0005, "momentum": 0.937}
    optimizer = Prompt.ask("[yellow]  Optimizer[/]", default="auto", choices=["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam"])
    cos_lr = Confirm.ask("[yellow]  Cosine LR scheduler?[/]", default=True)
    lr0 = FloatPrompt.ask("[yellow]  Initial learning rate[/]", default=defaults["lr0"])
    lrf = FloatPrompt.ask("[yellow]  Final LR ratio (lrf)[/]", default=defaults["lrf"])
    momentum = FloatPrompt.ask("[yellow]  Momentum[/]", default=defaults["momentum"])
    weight_decay = FloatPrompt.ask("[yellow]  Weight decay[/]", default=defaults["weight_decay"])
    warmup_epochs = FloatPrompt.ask("[yellow]  Warmup epochs[/]", default=3.0)
    warmup_momentum = FloatPrompt.ask("[yellow]  Warmup momentum[/]", default=0.8)
    warmup_bias_lr = FloatPrompt.ask("[yellow]  Warmup bias LR[/]", default=0.1)
    nbs = IntPrompt.ask("[yellow]  Nominal batch size (nbs)[/]", default=64)
    return {"optimizer": optimizer, "cos_lr": cos_lr, "lr0": lr0, "lrf": lrf, "momentum": momentum, "weight_decay": weight_decay, "warmup_epochs": warmup_epochs, "warmup_momentum": warmup_momentum, "warmup_bias_lr": warmup_bias_lr, "nbs": nbs}


def _configure_cls_caching_performance() -> dict:
    console.print()
    console.print("[bold]Caching & Performance:[/]")
    console.print()
    cache_choice = Prompt.ask("[yellow]  Dataset caching[/]", default="False", choices=["False", "ram", "disk"])
    cache = cache_choice if cache_choice != "False" else False
    amp = Confirm.ask("[yellow]  Mixed precision (AMP)?[/]", default=True)
    return {"cache": cache, "amp": amp}


def _configure_reproducibility() -> dict:
    console.print()
    console.print("[bold]Reproducibility:[/]")
    console.print()
    seed = IntPrompt.ask("[yellow]  Random seed[/]", default=0)
    deterministic = Confirm.ask("[yellow]  Deterministic mode?[/]", default=True)
    return {"seed": seed, "deterministic": deterministic}


def step_configure_training(dataset_info: dict) -> dict:
    console.print(Rule("[bold cyan]Step 2: Configure Training Parameters[/]"))
    console.print()
    model = _select_cls_model()
    console.print()
    preset_config = _select_cls_preset()
    preset_key = preset_config.pop("_preset_key")
    detected_device, device_desc = _detect_device()
    console.print()
    console.print(f"[bold]Detected device:[/] [green]{device_desc}[/]")
    override_device = Confirm.ask("[yellow]Use detected device?[/]", default=True)
    device = detected_device if override_device else Prompt.ask("[yellow]Enter device manually[/]", default=detected_device)

    if preset_key == "custom":
        console.print()
        console.print("[bold]Training Parameters:[/]")
        epochs = IntPrompt.ask("[yellow]  Epochs[/]", default=50)
        console.print("[dim]  Batch: 'auto' = auto-detect optimal, 'auto-70' = use 70% GPU, or integer[/]")
        batch_str = Prompt.ask("[yellow]  Batch size[/]", default="32")
        batch = _parse_batch_input(batch_str, device)
        if batch == -1 and device == "cpu":
            console.print("[yellow]  Auto-batch requires GPU. Using batch=16.[/]")
            batch = 16
        if (batch == -1 or isinstance(batch, float)) and _is_multi_gpu(device):
            gpu_count = _get_gpu_count(device)
            default_batch = 16 * gpu_count
            console.print(f"[yellow]  Auto-batch not supported for Multi-GPU training ({gpu_count} GPUs).[/]")
            console.print(f"[yellow]  Batch size must be an integer multiple of GPU count ({gpu_count}).[/]")
            batch = IntPrompt.ask(f"[yellow]  Enter batch size (multiple of {gpu_count})[/]", default=default_batch)
            if batch % gpu_count != 0:
                batch = (batch // gpu_count) * gpu_count
                console.print(f"[yellow]  Adjusted batch to {batch} (must be multiple of {gpu_count}).[/]")
        img_size = IntPrompt.ask("[yellow]  Image size[/]", default=224)
        patience = IntPrompt.ask("[yellow]  Early stopping patience[/]", default=15)
        dropout = FloatPrompt.ask("[yellow]  Dropout (0.0-1.0)[/]", default=0.0)
        workers = IntPrompt.ask("[yellow]  Dataloader workers[/]", default=8)
        config: dict = {"epochs": epochs, "batch": batch, "imgsz": img_size, "patience": patience, "workers": workers}
        if dropout > 0:
            config["dropout"] = dropout
    else:
        config = preset_config.copy()
        if config.get("batch") == -1 and device == "cpu":
            console.print("[yellow]Auto-batch requires GPU. Using batch=16.[/]")
            config["batch"] = 16
        batch_val = config.get("batch", 32)
        if (batch_val == -1 or isinstance(batch_val, float)) and _is_multi_gpu(device):
            gpu_count = _get_gpu_count(device)
            default_batch = 16 * gpu_count
            console.print(f"[yellow]Auto-batch not supported for Multi-GPU training ({gpu_count} GPUs).[/]")
            console.print(f"[yellow]Batch size must be an integer multiple of GPU count ({gpu_count}).[/]")
            config["batch"] = IntPrompt.ask(f"[yellow]Enter batch size (multiple of {gpu_count})[/]", default=default_batch)
            if config["batch"] % gpu_count != 0:
                config["batch"] = (config["batch"] // gpu_count) * gpu_count
                console.print(f"[yellow]Adjusted batch to {config['batch']} (must be multiple of {gpu_count}).[/]")
        console.print()
        if Confirm.ask("[yellow]Override any preset values?[/]", default=False):
            config["epochs"] = IntPrompt.ask("[yellow]  Epochs[/]", default=config.get("epochs", 50))
            config["imgsz"] = IntPrompt.ask("[yellow]  Image size[/]", default=config.get("imgsz", 224))
            config["patience"] = IntPrompt.ask("[yellow]  Patience[/]", default=config.get("patience", 15))
        config.setdefault("workers", 8)

    console.print()
    project_name = Prompt.ask("[yellow]Output project name[/]", default=f"train-{dataset_info['project']}")
    run_name = Prompt.ask("[yellow]Run name[/]", default=datetime.now().strftime("%Y%m%d_%H%M%S"))

    console.print()
    use_advanced = Confirm.ask("[yellow]Configure advanced options?[/]", default=False)
    advanced: dict = {}
    if use_advanced:
        console.print()
        console.print("[bold]Advanced Configuration Categories:[/]")
        console.print("  [dim]1.[/] Optimizer & Learning Rate")
        console.print("  [dim]2.[/] Augmentation")
        console.print("  [dim]3.[/] Caching & Performance")
        console.print("  [dim]4.[/] Reproducibility")
        console.print("  [dim]5.[/] All of the above")
        console.print()
        adv_choice = Prompt.ask("[yellow]Select categories (comma-separated)[/]", default="5")
        categories = [c.strip() for c in adv_choice.split(",")]
        if "1" in categories or "5" in categories:
            advanced.update(_configure_cls_optimizer_lr(model))
        if "2" in categories or "5" in categories:
            advanced.update(_configure_cls_augmentation())
        if "3" in categories or "5" in categories:
            advanced.update(_configure_cls_caching_performance())
        if "4" in categories or "5" in categories:
            advanced.update(_configure_reproducibility())
        console.print()
        console.print("[bold]Additional Options:[/]")
        freeze = IntPrompt.ask("[yellow]  Freeze layers (0=none)[/]", default=config.get("freeze", 0))
        dropout = FloatPrompt.ask("[yellow]  Dropout (0.0-1.0)[/]", default=config.get("dropout", 0.0))
        label_smoothing = FloatPrompt.ask("[yellow]  Label smoothing[/]", default=0.0)
        if freeze > 0:
            advanced["freeze"] = freeze
        if dropout > 0:
            advanced["dropout"] = dropout
        if label_smoothing > 0:
            advanced["label_smoothing"] = label_smoothing

    smart_defaults: dict = {"amp": True, "plots": True, "val": True, "exist_ok": False, "verbose": True}
    if _is_yolo26(model) and "lr0" not in advanced:
        size_key = _get_model_size_key(model)
        if size_key in YOLO26_DEFAULTS:
            yolo26_params = YOLO26_DEFAULTS[size_key]
            for param, value in yolo26_params.items():
                if param not in config and param not in advanced:
                    smart_defaults[param] = value
            console.print(f"[dim]Applied YOLO26-{size_key} optimized hyperparameters[/]")

    epochs = config.get("epochs", 50)
    if epochs > 100:
        smart_defaults["save_period"] = max(epochs // 10, 10)

    final_config = {"model": model + ".pt", "task": "classify", "data": dataset_info["path"], "device": device, "project": project_name, "name": run_name, **smart_defaults, **config, **advanced}
    final_config = {k: v for k, v in final_config.items() if v is not None}

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


def step_health_checks(config: dict, dataset_info: dict) -> None:
    console.print(Rule("[bold cyan]Pre-Training Health Checks[/]"))
    console.print()
    checks = _run_cls_health_checks(config, dataset_info)
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


def _launch_tensorboard(logdir: str) -> subprocess.Popen | None:
    if not shutil.which("tensorboard"):
        console.print("[yellow]TensorBoard not found. Install with: pip install tensorboard[/]")
        return None
    port = 6006
    try:
        proc = subprocess.Popen(["tensorboard", "--logdir", logdir, "--port", str(port), "--bind_all"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        console.print(f"[green]TensorBoard started at:[/] [bold]http://localhost:{port}[/]")
        console.print(f"[dim]Logging directory: {logdir}[/]")
        console.print()
        return proc
    except OSError as e:
        console.print(f"[yellow]Failed to start TensorBoard: {e}[/]")
        return None


def step_train(config: dict, resume_info: dict | None = None) -> dict:
    console.print(Rule("[bold cyan]Step 3: Training Model[/]"))
    console.print()
    use_tensorboard = Confirm.ask("[yellow]Launch TensorBoard for live monitoring?[/]", default=True)
    logdir = str(OUTPUT_DIR / "classify" / config.get("project", "")) if not resume_info else str(Path(resume_info["dir"]).parent)
    tb_process = None
    if use_tensorboard:
        from ultralytics import settings
        settings.update({"tensorboard": True})
        console.print("[dim]TensorBoard logging enabled (yolo settings tensorboard=True)[/]")
        tb_process = _launch_tensorboard(logdir)

    from ultralytics import YOLO
    start_time = time.time()

    if resume_info:
        console.print(f"[bold]Resuming from:[/] {resume_info['path']}")
        console.print(f"[dim]Completed: {resume_info['completed']}/{resume_info['total']} epochs[/]")
        model = YOLO(resume_info["path"])
        console.print("[bold]Resuming training...[/]")
        console.print()
        results = model.train(resume=True)
    else:
        console.print(f"[bold]Loading model:[/] {config['model']}")
        model = YOLO(config["model"])
        console.print("[bold]Starting classification training...[/]")
        console.print(f"[dim]Output directory: {OUTPUT_DIR}/classify/{config['project']}/{config['name']}[/]")
        console.print()
        exclude_keys = {"task"}
        train_args = {k: v for k, v in config.items() if k not in exclude_keys}
        results = model.train(**train_args)

    train_dir = Path(str(model.trainer.save_dir)) if model.trainer else Path("runs/classify")
    elapsed = time.time() - start_time
    console.print()
    console.print("[bold green]Training complete![/]")
    console.print(f"[dim]Results saved to: {train_dir}[/]")
    console.print(f"[dim]Total time: {elapsed/60:.1f} minutes[/]")

    if tb_process and tb_process.poll() is None:
        console.print()
        keep_tb = Confirm.ask("[yellow]TensorBoard is still running. Keep it open?[/]", default=False)
        if not keep_tb:
            tb_process.terminate()
            tb_process.wait(timeout=5)
            console.print("[dim]TensorBoard stopped.[/]")
    console.print()
    return {"results": results, "train_dir": str(train_dir), "model_path": str(train_dir / "weights" / "best.pt"), "elapsed_seconds": elapsed}


def step_validate(train_info: dict, config: dict) -> object | None:
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
    # Resolve dataset path: prefer config, fallback to args.yaml from training output
    data_path = config.get("data")
    if not data_path:
        args_yaml = Path(train_info["train_dir"]) / "args.yaml"
        if args_yaml.exists():
            with open(args_yaml) as f:
                train_args = yaml.safe_load(f) or {}
            data_path = train_args.get("data")
    val_args: dict = {"data": data_path, "imgsz": config.get("imgsz", 224), "batch": config.get("batch", 32) if config.get("batch", 32) != -1 else 32, "device": config.get("device", "cpu"), "plots": True, "verbose": True}
    metrics = model.val(**val_args)
    console.print()
    val_table = Table(title="Classification Validation Results", box=box.ROUNDED)
    val_table.add_column("Metric", style="cyan")
    val_table.add_column("Value", style="green")
    try:
        if hasattr(metrics, "top1"):
            val_table.add_row("Top-1 Accuracy", f"{metrics.top1:.4f}")
        if hasattr(metrics, "top5"):
            val_table.add_row("Top-5 Accuracy", f"{metrics.top5:.4f}")
        console.print(val_table)
    except Exception as e:
        console.print(f"[yellow]Could not parse validation metrics: {e}[/]")
    console.print()
    return metrics


def _find_cls_best_epoch(csv_path: Path) -> dict | None:
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return None
        top1_col = None
        for col in rows[0].keys():
            col_stripped = col.strip()
            if "metrics/accuracy_top1" in col_stripped or "top1" in col_stripped.lower():
                top1_col = col
                break
        if not top1_col:
            return None
        best_row = max(rows, key=lambda r: float(r.get(top1_col, "0").strip() or "0"))
        return best_row
    except Exception:
        return None


def _display_cls_metrics_from_csv(csv_path: Path) -> None:
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return
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
        best_row = _find_cls_best_epoch(csv_path)
        if best_row and best_row != last_row:
            best_table = Table(title="Best Epoch Metrics (by Top-1 Accuracy)", box=box.ROUNDED)
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


def _generate_cls_training_summary_paragraph(config: dict, dataset_info: dict, train_info: dict, train_dir: Path) -> tuple[str, list[str]]:
    model = config.get("model", "unknown")
    epochs = config.get("epochs", 0)
    imgsz = config.get("imgsz", 224)
    batch = config.get("batch", -1)
    elapsed = train_info.get("elapsed_seconds", 0)
    dataset_name = f"{dataset_info['project']} v{dataset_info.get('version', 'N/A')}"

    final_metrics: dict[str, float] = {}
    best_metrics: dict[str, float] = {}
    results_csv = train_dir / "results.csv"
    if results_csv.exists():
        try:
            with open(results_csv) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                for key, value in rows[-1].items():
                    try:
                        final_metrics[key.strip()] = float(value.strip())
                    except (ValueError, AttributeError):
                        pass
                top1_col = None
                for col in rows[0].keys():
                    col_stripped = col.strip()
                    if "metrics/accuracy_top1" in col_stripped or "top1" in col_stripped.lower():
                        top1_col = col
                        break
                if top1_col:
                    best_row = max(rows, key=lambda r: float(r.get(top1_col, "0").strip() or "0"))
                    for key, value in best_row.items():
                        try:
                            best_metrics[key.strip()] = float(value.strip())
                        except (ValueError, AttributeError):
                            pass
        except Exception:
            pass

    best_top1: float = best_metrics.get("metrics/accuracy_top1", 0.0)
    best_top5: float = best_metrics.get("metrics/accuracy_top5", 0.0)
    best_epoch: float = best_metrics.get("epoch", 0.0)
    final_epoch: float = final_metrics.get("epoch", float(epochs))
    train_loss: float = final_metrics.get("train/loss", 0.0)
    val_loss: float = final_metrics.get("val/loss", 0.0)

    if best_top1 >= 0.95:
        quality, quality_emoji = "sangat baik", "[bold green]"
    elif best_top1 >= 0.90:
        quality, quality_emoji = "baik", "[green]"
    elif best_top1 >= 0.80:
        quality, quality_emoji = "cukup", "[yellow]"
    elif best_top1 >= 0.70:
        quality, quality_emoji = "kurang", "[red]"
    else:
        quality, quality_emoji = "sangat rendah", "[bold red]"

    if elapsed > 0:
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        time_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m {int(elapsed % 60)}s"
    else:
        time_str = "N/A"

    summary = (f"Training model [bold]{model}[/] pada dataset [bold]{dataset_name}[/] telah selesai dalam waktu [bold]{time_str}[/] dengan total [bold]{int(final_epoch)}[/] epoch. Performa terbaik dicapai pada epoch [bold]{int(best_epoch)}[/] dengan Top-1 Accuracy = [bold]{best_top1:.4f}[/] dan Top-5 Accuracy = [bold]{best_top5:.4f}[/]. Secara keseluruhan, hasil training ini dinilai {quality_emoji}{quality}[/].")

    recommendations = []
    if final_epoch > 0 and best_epoch > 0:
        epoch_ratio = best_epoch / final_epoch
        if epoch_ratio < 0.2 and final_epoch > 10:
            recommendations.append(f"[yellow]Overfitting terdeteksi:[/] Performa terbaik di epoch awal ({int(best_epoch)}/{int(final_epoch)}). Model mulai overfit setelahnya. Coba kurangi epoch, tambah augmentasi, naikkan dropout, atau gunakan early stopping (patience lebih kecil).")
    if best_top1 < 0.5:
        recommendations.append("[yellow]Underfitting terdeteksi:[/] Top-1 accuracy sangat rendah. Kemungkinan penyebab: dataset terlalu kecil, label kurang akurat, atau model terlalu kecil untuk task ini. Coba: (1) tambah data training, (2) periksa kualitas gambar per kelas, (3) gunakan model lebih besar (m/l/x), (4) tambah epoch.")
    if train_loss > 0 and val_loss > 0:
        loss_ratio = val_loss / train_loss
        if loss_ratio > 2.0:
            recommendations.append(f"[yellow]Gap train/val loss besar:[/] val_loss ({val_loss:.3f}) jauh lebih tinggi dari train_loss ({train_loss:.3f}). Ini menandakan overfitting. Coba: tambah augmentasi (heavy), naikkan dropout, atau tambah data.")
    if best_top1 > 0 and best_top5 > 0:
        gap = best_top5 - best_top1
        if gap > 0.15 and best_top1 < 0.8:
            recommendations.append(f"[yellow]Gap Top-1 vs Top-5 besar ({gap:.2f}):[/] Model sering bingung antara kelas yang mirip. Coba: periksa apakah ada kelas yang visual-nya mirip, tambah data per kelas, atau gunakan image size lebih besar.")
    dropout = config.get("dropout", 0.0)
    if best_top1 < 0.7 and dropout == 0.0:
        recommendations.append("[cyan]Dropout belum digunakan:[/] Coba tambahkan dropout (0.1-0.3) untuk mengurangi overfitting.")
    if epochs <= 20 and best_top1 < 0.8:
        recommendations.append(f"[cyan]Epoch terlalu sedikit:[/] Dengan hanya {epochs} epoch, model mungkin belum konvergen. Coba naikkan ke 50-150 epoch untuk hasil lebih baik.")
    if imgsz < 224 and best_top1 < 0.7:
        recommendations.append(f"[cyan]Image size kecil ({imgsz}):[/] Untuk klasifikasi, gunakan minimal 224. Coba naikkan ke 224 atau 320.")
    if batch == -1:
        recommendations.append("[dim]Auto-batch digunakan \u2014 GPU memory dioptimalkan secara otomatis.[/]")
    if best_top1 >= 0.8 and not recommendations:
        recommendations.append("[green]Model sudah memiliki performa yang baik![/] Untuk peningkatan lebih lanjut, coba: fine-tune dengan learning rate lebih kecil, tambah data, atau gunakan Test-Time Augmentation (TTA) saat inference.")
    elif best_top1 >= 0.95:
        recommendations.insert(0, "[green]Performa sangat baik![/] Model siap untuk deployment. Pertimbangkan export ke ONNX/TensorRT untuk inference lebih cepat.")
    return summary, recommendations


def step_summary(train_info: dict, config: dict, dataset_info: dict) -> None:
    console.print(Rule("[bold cyan]Step 5: Training Summary[/]"))
    console.print()
    train_dir = Path(train_info["train_dir"])
    overview = Table(title="Training Overview", box=box.ROUNDED)
    overview.add_column("Property", style="cyan")
    overview.add_column("Value", style="green")
    overview.add_row("Model", config.get("model", "N/A"))
    overview.add_row("Dataset", f"{dataset_info['project']} v{dataset_info.get('version', 'N/A')}")
    overview.add_row("Task", "classify")
    overview.add_row("Epochs", str(config.get("epochs", "N/A")))
    overview.add_row("Batch Size", str(config.get("batch", "N/A")))
    overview.add_row("Image Size", str(config.get("imgsz", "N/A")))
    overview.add_row("Device", str(config.get("device", "N/A")))
    overview.add_row("Best Model", train_info["model_path"])
    elapsed = train_info.get("elapsed_seconds", 0)
    if elapsed > 0:
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        if hours > 0:
            overview.add_row("Training Time", f"{hours}h {minutes}m")
        else:
            overview.add_row("Training Time", f"{minutes}m {int(elapsed % 60)}s")
    best_pt = Path(train_info["model_path"])
    if best_pt.exists():
        overview.add_row("Model Size", _format_size(best_pt.stat().st_size))
    console.print(overview)
    console.print()

    try:
        summary_text, recommendations = _generate_cls_training_summary_paragraph(config, dataset_info, train_info, train_dir)
        console.print(Panel(summary_text, title="[bold]Ringkasan Training[/]", border_style="bright_blue", padding=(1, 2)))
        console.print()
        if recommendations:
            rec_text = "\n".join(f"  {i}. {rec}" for i, rec in enumerate(recommendations, 1))
            console.print(Panel(rec_text, title="[bold]Rekomendasi & Saran[/]", border_style="yellow", padding=(1, 2)))
            console.print()
    except Exception:
        pass

    results_csv = train_dir / "results.csv"
    if results_csv.exists():
        _display_cls_metrics_from_csv(results_csv)
    else:
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

    if not train_dir.exists():
        console.print("[yellow]Training directory not found \u2014 cannot list output files.[/]")
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
        console.print(f"  [dim]{i}.[/] [bold]{key}[/] \u2014 {EXPORT_FORMATS[key]['desc']}")
    console.print()
    console.print("[dim]Enter numbers separated by commas (e.g., 1,2,4)[/]")
    selection = Prompt.ask("[yellow]Select formats[/]", default="1")
    selected_indices = [int(s.strip()) for s in selection.split(",") if s.strip().isdigit()]
    selected_formats = [format_keys[i - 1] for i in selected_indices if 1 <= i <= len(format_keys)]
    if not selected_formats:
        console.print("[yellow]No valid formats selected. Skipping export.[/]")
        return []
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
            export_args: dict = {"format": fmt, "half": half, "dynamic": dynamic, "imgsz": config.get("imgsz", 224)}
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
            fmt_name = p.suffix.lstrip(".")
            size = _format_size(p.stat().st_size) if p.exists() else "N/A"
            export_table.add_row(fmt_name, str(p.name), size)
        console.print(export_table)
        console.print()
    return exported_paths


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
    include_weights = Confirm.ask("[yellow]Include model weights (best.pt & last.pt)?[/]", default=True)
    include_plots = Confirm.ask("[yellow]Include plots and visualizations?[/]", default=True)
    include_all = Confirm.ask("[yellow]Include all other files (csv, args, etc.)?[/]", default=True)
    console.print()
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
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
                if not include_all and not any(rel_str.endswith(ext) for ext in [".pt", ".png", ".jpg", ".csv", ".yaml"]):
                    continue
                zf.write(item, arcname=rel_str)
        progress.update(zip_task, description="[green]Zip created!")
    zip_size = _format_size(zip_path.stat().st_size)
    console.print()
    console.print(Panel(f"[bold green]Exported:[/] {zip_path}\n[bold green]Size:[/] {zip_size}", title="Export Complete", border_style="green"))
    console.print()
    return str(zip_path)


def mode_tune(dataset_info: dict) -> None:
    console.print(Rule("[bold magenta]Hyperparameter Tuning Mode (Classification)[/]"))
    console.print()
    console.print("[dim]This mode uses Ultralytics' built-in hyperparameter evolution[/]")
    console.print("[dim]to find optimal training parameters for your classification dataset.[/]")
    console.print()
    model_name = _select_cls_model()
    from ultralytics import YOLO
    model = YOLO(model_name + ".pt")
    iterations = IntPrompt.ask("[yellow]Tuning iterations[/]", default=300)
    tune_epochs = IntPrompt.ask("[yellow]Epochs per iteration[/]", default=30)
    imgsz = IntPrompt.ask("[yellow]Image size[/]", default=224)
    detected_device, device_desc = _detect_device()
    console.print(f"[bold]Device:[/] [green]{device_desc}[/]")
    console.print()
    console.print("[bold]Starting hyperparameter tuning...[/]")
    console.print(f"[dim]This will run {iterations} iterations x {tune_epochs} epochs each.[/]")
    console.print(f"[dim]Estimated time: {iterations * tune_epochs * 0.5:.0f}+ minutes (varies by hardware)[/]")
    console.print()
    if not Confirm.ask("[yellow]Proceed?[/]", default=True):
        return
    model.tune(data=dataset_info["path"], task="classify", epochs=tune_epochs, iterations=iterations, imgsz=imgsz, device=detected_device, plots=True, val=True)
    console.print()
    console.print("[bold green]Tuning complete![/]")
    console.print("[dim]Best hyperparameters saved to runs/classify/tune/best_hyperparameters.yaml[/]")
    console.print()


def _load_training_config() -> dict | None:
    yaml_files = list(Path(".").glob("config-*.yaml")) + list(Path(".").glob("training_config*.yaml"))
    if not yaml_files:
        return None
    console.print("[bold]Found saved configurations:[/]")
    for i, f in enumerate(yaml_files, 1):
        console.print(f"  [dim]{i}.[/] {f.name}")
    console.print()
    choice = Prompt.ask("[yellow]Load a saved config?[/]", choices=["skip"] + [str(i) for i in range(1, len(yaml_files) + 1)], default="skip")
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


def main():
    console.print()
    console.print(Panel(Text.from_markup("[bold white]YOLO Classification Training Wizard[/]\n[dim]Train YOLO classification models with Roboflow datasets \u2014 optimized for best practices[/]\n\n[cyan]1.[/] Download/Select Classification Dataset\n[cyan]2.[/] Configure Training (with presets)\n[cyan]3.[/] Train Model\n[cyan]4.[/] Validate Model\n[cyan]5.[/] View Training Summary\n[cyan]6.[/] Export Model (ONNX/TensorRT/etc.)\n[cyan]7.[/] Export Results (Zip)\n\n[dim]Features: Roboflow paste-to-train, Auto-batch, Resume, YOLO26 defaults,[/]\n[dim]Augmentation presets, Health checks, Model export, HP tuning mode[/]"), border_style="bright_blue", padding=(1, 2)))
    console.print()

    if "--tune" in sys.argv:
        console.print("[bold magenta]Entering Hyperparameter Tuning Mode[/]")
        console.print()
        dataset_info = step_download_dataset()
        mode_tune(dataset_info)
        return

    try:
        resume_info = step_check_resume()
        exported = []
        zip_path = ""

        if resume_info:
            train_info = step_train({}, resume_info=resume_info)
            args_yaml = Path(resume_info["dir"]) / "args.yaml"
            config = {}
            if args_yaml.exists():
                with open(args_yaml) as f:
                    config = yaml.safe_load(f) or {}
            dataset_info = {"project": config.get("name", "resumed"), "version": "N/A", "classes": [], "splits": {}, "path": config.get("data", "")}
            step_validate(train_info, config)
            step_summary(train_info, config, dataset_info)
            exported = step_export_model(train_info, config)
            zip_path = step_zip_results(train_info, config)
        else:
            saved_config = _load_training_config()
            dataset_info = step_download_dataset()
            if saved_config:
                config = saved_config
                config["data"] = dataset_info["path"]
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

        summary_lines = ["[bold green]All done! Your YOLO classification model has been trained and exported.[/]\n", f"[cyan]Best model:[/] {train_info['model_path']}"]
        if exported:
            summary_lines.append(f"[cyan]Exported formats:[/] {len(exported)}")
        if zip_path:
            summary_lines.append(f"[cyan]Zip export:[/] {zip_path}")
        console.print(Panel("\n".join(summary_lines), title="\U0001f52c Wizard Complete", border_style="bright_green", padding=(1, 2)))

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
