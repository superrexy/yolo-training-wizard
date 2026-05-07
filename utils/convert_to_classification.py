#!/usr/bin/env python3
"""Convert YOLO detection dataset to YOLO classification format.

Crops bounding boxes from detection images and organizes them into
class-based folder structure suitable for YOLO classification training.

Usage:
    python utils/convert_to_classification.py -i datasets/my-project -o datasets/my-project-cls
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image
from PIL.Image import Resampling
from rich import box
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.progress import track

console = Console()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

SPLIT_NAME_MAP = {
    "train": "train",
    "valid": "val",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert YOLO detection dataset to classification format.",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to detection dataset (must contain data.yaml)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Path for classification output",
    )
    parser.add_argument(
        "--margin", "-m",
        type=float,
        default=0.15,
        help="Margin as fraction of bbox dimensions (default: 0.15)",
    )
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=224,
        help="Target crop size in pixels (default: 224)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=32,
        help="Minimum crop dimension before filtering (default: 32)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output directory",
    )
    return parser.parse_args()


def validate_dataset(input_path: Path) -> tuple[dict[int, str], list[tuple[str, str]]]:
    """Validate dataset structure and return class names and available splits.

    Returns:
        Tuple of (names_dict, list_of_split_tuples) where each split tuple
        is (input_split_name, output_split_name).
    """
    yaml_path = input_path / "data.yaml"
    if not yaml_path.exists():
        console.print("[bold red]Error:[/] data.yaml not found in " + str(input_path))
        sys.exit(1)

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if data is None or "names" not in data:
        console.print("[bold red]Error:[/] data.yaml missing 'names' key")
        sys.exit(1)

    names = data["names"]
    if not isinstance(names, dict):
        console.print("[bold red]Error:[/] 'names' must be a dict (int -> str mapping)")
        sys.exit(1)

    splits: list[tuple[str, str]] = []
    for split_dir, output_name in SPLIT_NAME_MAP.items():
        split_path = input_path / split_dir
        if split_path.is_dir():
            images_dir = split_path / "images"
            labels_dir = split_path / "labels"
            if not images_dir.is_dir() or not labels_dir.is_dir():
                console.print(
                    "[bold red]Error:[/] Split '"
                    + split_dir
                    + "' must have images/ and labels/ subdirectories"
                )
                sys.exit(1)
            splits.append((split_dir, output_name))

    if not splits:
        console.print(
            "[bold red]Error:[/] No split directories found (expected train/, valid/, or test/)"
        )
        sys.exit(1)

    console.print(
        "[green]Found "
        + str(len(names))
        + " classes and "
        + str(len(splits))
        + " splits[/]"
    )
    return names, splits


def letterbox(
    image: Image.Image,
    target_size: int,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> Image.Image:
    """Resize image preserving aspect ratio with padding.

    Scales the image so the longest side fits target_size, then centers
    it on a square canvas filled with the given color.
    """
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = image.resize((new_w, new_h), Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_size, target_size), fill)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    return canvas


def process_split(
    input_path: Path,
    output_path: Path,
    split_name: str,
    output_split_name: str,
    names: dict[int, str],
    margin: float,
    target_size: int,
    min_size: int,
) -> tuple[dict[str, int], int, int]:
    """Process one split. Returns (crop_counts_dict, skipped, filtered)."""
    images_dir = input_path / split_name / "images"
    labels_dir = input_path / split_name / "labels"

    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    crop_counts: dict[str, int] = {name: 0 for name in names.values()}
    skipped = 0
    filtered = 0

    for img_path in track(image_files, description="  " + output_split_name):
        label_path = labels_dir / (img_path.stem + ".txt")

        if not label_path.exists():
            skipped += 1
            continue

        lines = label_path.read_text().strip().splitlines()
        if not lines:
            skipped += 1
            continue

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            if class_id not in names:
                continue

            class_name = names[class_id]

            x_center = float(parts[1]) * img_w
            y_center = float(parts[2]) * img_h
            bbox_w = float(parts[3]) * img_w
            bbox_h = float(parts[4]) * img_h

            margin_x = margin * bbox_w
            margin_y = margin * bbox_h

            x1 = max(0, x_center - bbox_w / 2 - margin_x)
            y1 = max(0, y_center - bbox_h / 2 - margin_y)
            x2 = min(img_w, x_center + bbox_w / 2 + margin_x)
            y2 = min(img_h, y_center + bbox_h / 2 + margin_y)

            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w < min_size or crop_h < min_size:
                filtered += 1
                continue

            crop = img.crop((x1, y1, x2, y2))
            crop = letterbox(crop, target_size)

            out_dir = output_path / output_split_name / class_name
            filename = img_path.stem + "_" + class_name + "_" + str(idx) + ".jpg"
            crop.save(out_dir / filename, "JPEG", quality=95)

            crop_counts[class_name] += 1

    return crop_counts, skipped, filtered


def main() -> None:
    """Entry point for detection-to-classification conversion."""
    try:
        args = parse_args()

        console.print(Rule("[bold cyan]Converting Detection to Classification[/]"))

        names, splits = validate_dataset(args.input)

        if args.output.exists():
            if not args.overwrite:
                console.print(
                    "[bold red]Error:[/] Output directory already exists: "
                    + str(args.output)
                )
                console.print("Use [bold]--overwrite[/] to replace it")
                sys.exit(1)
            shutil.rmtree(args.output)

        for _, output_split_name in splits:
            for class_name in names.values():
                (args.output / output_split_name / class_name).mkdir(
                    parents=True, exist_ok=True
                )

        all_counts: dict[str, dict[str, int]] = {}
        total_skipped = 0
        total_filtered = 0

        for split_name, output_split_name in splits:
            console.print("\n[bold]Processing split:[/] " + output_split_name)
            crop_counts, skipped, filtered = process_split(
                input_path=args.input,
                output_path=args.output,
                split_name=split_name,
                output_split_name=output_split_name,
                names=names,
                margin=args.margin,
                target_size=args.size,
                min_size=args.min_size,
            )
            all_counts[output_split_name] = crop_counts
            total_skipped += skipped
            total_filtered += filtered

        console.print()
        table = Table(title="Conversion Results", box=box.ROUNDED)
        table.add_column("Split", style="cyan")
        table.add_column("Class", style="green")
        table.add_column("Crops", style="yellow", justify="right")

        for split_label, counts in all_counts.items():
            split_total = 0
            for class_name, count in sorted(counts.items()):
                table.add_row(split_label, class_name, str(count))
                split_total += count
            table.add_row(
                split_label, "[bold]Total[/]", "[bold]" + str(split_total) + "[/]"
            )
            table.add_section()

        console.print(table)

        console.print("[dim]Skipped (no annotations):[/] " + str(total_skipped))
        console.print("[dim]Filtered (below min size):[/] " + str(total_filtered))
        console.print("[bold green]Output:[/] " + str(args.output.resolve()))

    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user.[/]")
        sys.exit(0)
    except Exception as e:
        console.print("[bold red]Error:[/] " + str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
