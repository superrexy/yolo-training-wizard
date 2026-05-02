import os
import sys
import csv
import json
import platform
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
from rich.rule import Rule
from rich.text import Text
from rich import box

console = Console()

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
        return data.get("names", [])
    except Exception:
        return []


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
    Priority: CUDA (all GPUs) → MPS (Apple Silicon) → CPU
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
            return "mps", "MPS (Apple Silicon — torch check skipped)"

    return "cpu", "CPU (no GPU detected)"


# ─── Step 1 ───────────────────────────────────────────────────────────────────


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
        default="yolov8",
        choices=["yolov8", "yolov5", "yolov7", "yolov9", "coco", "voc"],
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
        table.add_row("Classes", ", ".join(classes))
        table.add_row("Num Classes", str(len(classes)))
    console.print(table)
    console.print()

    return info


# ─── Step 2 ───────────────────────────────────────────────────────────────────


def step_configure_training(dataset_info: dict) -> dict:
    console.print(Rule("[bold cyan]Step 2: Configure Training Parameters[/]"))
    console.print()

    console.print("[bold]Available models:[/]")
    for i, model in enumerate(YOLO_MODELS, 1):
        console.print(f"  [dim]{i:2d}.[/] {model}")
    console.print()

    model_idx = IntPrompt.ask(
        "[yellow]Select model number[/]",
        default=1,
        choices=[str(i) for i in range(1, len(YOLO_MODELS) + 1)],
    )
    model = YOLO_MODELS[model_idx - 1]

    task = Prompt.ask("[yellow]Task[/]", default="detect", choices=TASKS)

    console.print()
    console.print("[bold]Training Parameters:[/]")
    epochs = IntPrompt.ask("[yellow]  Epochs[/]", default=100)
    batch_size = IntPrompt.ask("[yellow]  Batch size[/]", default=16)
    img_size = IntPrompt.ask("[yellow]  Image size[/]", default=640)
    lr0 = FloatPrompt.ask("[yellow]  Initial learning rate[/]", default=0.01)
    patience = IntPrompt.ask("[yellow]  Early stopping patience[/]", default=50)
    workers = IntPrompt.ask("[yellow]  Dataloader workers[/]", default=8)

    detected_device, device_desc = _detect_device()
    console.print(f"[bold]  Detected device:[/] [green]{device_desc}[/]")
    override_device = Confirm.ask("[yellow]  Use detected device?[/]", default=True)
    if override_device:
        device = detected_device
    else:
        device = Prompt.ask("[yellow]  Enter device manually[/]", default=detected_device)

    project_name = Prompt.ask(
        "[yellow]  Output project name[/]",
        default=f"train-{dataset_info['project']}",
    )
    run_name = Prompt.ask(
        "[yellow]  Run name[/]",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    console.print()
    use_advanced = Confirm.ask("[yellow]Configure advanced options?[/]", default=False)

    advanced = {}
    if use_advanced:
        advanced["optimizer"] = Prompt.ask(
            "[yellow]  Optimizer[/]",
            default="auto",
            choices=["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam"],
        )
        advanced["cos_lr"] = Confirm.ask(
            "[yellow]  Use cosine LR scheduler?[/]", default=False
        )
        advanced["augment"] = Confirm.ask(
            "[yellow]  Enable augmentation?[/]", default=True
        )
        advanced["freeze"] = IntPrompt.ask(
            "[yellow]  Freeze layers (0=none)[/]", default=0
        )
        advanced["dropout"] = FloatPrompt.ask("[yellow]  Dropout[/]", default=0.0)

    config = {
        "model": model,
        "task": task,
        "data": dataset_info["data_yaml"],
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": img_size,
        "lr0": lr0,
        "patience": patience,
        "workers": workers,
        "device": device,
        "project": str(OUTPUT_DIR / project_name),
        "name": run_name,
        **advanced,
    }

    console.print()
    table = Table(title="Training Configuration", box=box.ROUNDED)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    for key, value in config.items():
        table.add_row(key, str(value))
    console.print(table)
    console.print()

    if not Confirm.ask("[yellow]Proceed with training?[/]", default=True):
        console.print("[red]Training cancelled by user.[/]")
        sys.exit(0)

    return config


# ─── Step 3 ───────────────────────────────────────────────────────────────────


def step_train(config: dict) -> dict:
    console.print(Rule("[bold cyan]Step 3: Training Model[/]"))
    console.print()

    from ultralytics import YOLO

    console.print(f"[bold]Loading model:[/] {config['model']}")
    model = YOLO(config["model"])

    console.print("[bold]Starting training...[/]")
    console.print(f"[dim]Output directory: {config['project']}/{config['name']}[/]")
    console.print()

    train_args = {k: v for k, v in config.items() if k != "task"}
    results = model.train(**train_args)

    train_dir = Path(config["project"]) / config["name"]

    console.print()
    console.print("[bold green]Training complete![/]")
    console.print(f"[dim]Results saved to: {train_dir}[/]")
    console.print()

    return {
        "results": results,
        "train_dir": str(train_dir),
        "model_path": str(train_dir / "weights" / "best.pt"),
    }


# ─── Step 4 ───────────────────────────────────────────────────────────────────


def _display_metrics_from_csv(csv_path: Path) -> None:
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

    except Exception as e:
        console.print(f"[yellow]Could not parse results.csv: {e}[/]")


def step_summary(train_info: dict, config: dict, dataset_info: dict) -> None:
    console.print(Rule("[bold cyan]Step 4: Training Summary[/]"))
    console.print()

    train_dir = Path(train_info["train_dir"])

    overview = Table(title="Training Overview", box=box.ROUNDED)
    overview.add_column("Property", style="cyan")
    overview.add_column("Value", style="green")
    overview.add_row("Model", config["model"])
    overview.add_row("Dataset", f"{dataset_info['project']} v{dataset_info['version']}")
    overview.add_row("Task", config.get("task", "detect"))
    overview.add_row("Epochs", str(config["epochs"]))
    overview.add_row("Batch Size", str(config["batch"]))
    overview.add_row("Image Size", str(config["imgsz"]))
    overview.add_row("Device", str(config["device"]))
    overview.add_row("Best Model", train_info["model_path"])
    console.print(overview)
    console.print()

    results_csv = train_dir / "results.csv"
    if results_csv.exists():
        _display_metrics_from_csv(results_csv)

    files_table = Table(title="Output Files", box=box.ROUNDED)
    files_table.add_column("File", style="cyan")
    files_table.add_column("Size", style="green")

    for item in sorted(train_dir.rglob("*")):
        if item.is_file():
            size_str = _format_size(item.stat().st_size)
            files_table.add_row(str(item.relative_to(train_dir)), size_str)

    console.print(files_table)
    console.print()


# ─── Step 5 ───────────────────────────────────────────────────────────────────


def step_zip_results(train_info: dict, config: dict) -> str:
    console.print(Rule("[bold cyan]Step 5: Export Results (Zip)[/]"))
    console.print()

    train_dir = Path(train_info["train_dir"])
    zip_name = f"{config['name']}_results.zip"
    zip_path = train_dir.parent / zip_name

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


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                "[bold white]YOLO Training Wizard[/]\n"
                "[dim]Train YOLO models with Roboflow datasets in 5 easy steps[/]\n\n"
                "[cyan]1.[/] Download Dataset from Roboflow\n"
                "[cyan]2.[/] Configure Training Parameters\n"
                "[cyan]3.[/] Train Model\n"
                "[cyan]4.[/] View Training Summary\n"
                "[cyan]5.[/] Export Results (Zip)"
            ),
            border_style="bright_blue",
            padding=(1, 2),
        )
    )
    console.print()

    try:
        dataset_info = step_download_dataset()
        config = step_configure_training(dataset_info)
        train_info = step_train(config)
        step_summary(train_info, config, dataset_info)
        zip_path = step_zip_results(train_info, config)

        console.print(
            Panel(
                "[bold green]All done! Your YOLO model has been trained and exported.[/]\n\n"
                f"[cyan]Best model:[/] {train_info['model_path']}\n"
                f"[cyan]Zip export:[/] {zip_path}",
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
