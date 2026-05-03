# YOLO Training Wizard

CLI wizard interaktif untuk training model YOLO dengan dataset dari Roboflow — dioptimalkan dengan best practices dan smart defaults.

## Features

- **Roboflow Integration** — Paste snippet langsung dari Roboflow untuk download dataset
- **Training Presets** — Quick Test, Balanced, Maximum Quality, Small Objects, Fine-Tune
- **Auto-Batch** — Otomatis mendeteksi batch size optimal berdasarkan GPU memory
- **YOLO26 Smart Defaults** — Hyperparameter yang sudah dioptimalkan per model size (n/s/m/l/x)
- **Augmentation Presets** — None, Light, Medium, Heavy, atau Custom
- **Resume Training** — Otomatis mendeteksi training yang terinterupsi
- **Health Checks** — Validasi disk space, dataset, dan konfigurasi sebelum training
- **Model Export** — ONNX, TensorRT, CoreML, TFLite, OpenVINO, NCNN, TorchScript
- **Hyperparameter Tuning** — Mode otomatis untuk mencari parameter optimal
- **Training Summary** — Analisis performa dengan rekomendasi perbaikan

## Supported Models

| Generation      | Variants                                    |
| --------------- | ------------------------------------------- |
| YOLO26 (Latest) | yolo26n, yolo26s, yolo26m, yolo26l, yolo26x |
| YOLO12          | yolo12n, yolo12s, yolo12m, yolo12l, yolo12x |
| YOLO11          | yolo11n, yolo11s, yolo11m, yolo11l, yolo11x |
| YOLOv8          | yolov8n, yolov8s, yolov8m, yolov8l, yolov8x |

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (recommended package manager)

## Penggunaan

### Local (Terminal)

```bash
# Clone repository
git clone https://github.com/superrexy/yolo-training-wizard.git
cd yolo-training-wizard

# Jalankan wizard (uv akan otomatis install dependencies)
MPLBACKEND=Agg uv run main.py
```

> **Catatan:** `MPLBACKEND=Agg` diperlukan untuk mencegah matplotlib membuka GUI window saat training, terutama di environment tanpa display (server, SSH, headless).

### Google Colab

```python
# Cell 1: Install uv
!curl -LsSf https://astral.sh/uv/install.sh | sh
!source $HOME/.local/bin/env
```

```python
# Cell 2: Clone dan jalankan
!git clone https://github.com/superrexy/yolo-training-wizard.git
%cd yolo-training-wizard
```

```python
# Cell 3: Jalankan wizard
%%shell
MPLBACKEND=Agg uv run main.py
```

> **Catatan untuk Colab:**
>
> - Gunakan `%%shell` magic agar interactive input (Rich prompts) berfungsi dengan benar
> - `MPLBACKEND=Agg` wajib karena Colab tidak memiliki display untuk matplotlib GUI
> - Pastikan runtime menggunakan GPU (Runtime → Change runtime type → T4 GPU) untuk training yang lebih cepat

### Hyperparameter Tuning Mode

```bash
# Local
MPLBACKEND=Agg uv run main.py --tune

# Google Colab
%%shell
MPLBACKEND=Agg uv run main.py --tune
```

Mode ini menggunakan Ultralytics built-in hyperparameter evolution untuk mencari parameter training optimal secara otomatis.

### Verbose Mode (Debug)

```bash
MPLBACKEND=Agg uv run main.py --verbose
```

Menampilkan full stack trace jika terjadi error.

## Konfigurasi

### Training Presets

Wizard menyediakan preset yang sudah dioptimalkan:

| Preset              | Epochs | Image Size | Patience | Deskripsi                                                |
| ------------------- | ------ | ---------- | -------- | -------------------------------------------------------- |
| **Quick Test**      | 10     | 320        | 5        | Validasi cepat bahwa semuanya berjalan                   |
| **Balanced**        | 100    | 640        | 50       | Kualitas baik dengan waktu training wajar                |
| **Maximum Quality** | 300    | 640        | 100      | Hasil terbaik — training lama (+ AdamW, label smoothing) |
| **Small Objects**   | 200    | 1280       | 80       | Optimized untuk deteksi objek kecil (+ multi-scale)      |
| **Fine-Tune**       | 50     | 640        | 20       | Fine-tune pretrained model (frozen backbone, lr rendah)  |
| **Custom**          | -      | -          | -        | Konfigurasi manual semua parameter                       |

### Augmentation Presets

| Preset     | Deskripsi                                                 | Mosaic  | Mixup   | Flip LR |
| ---------- | --------------------------------------------------------- | ------- | ------- | ------- |
| **None**   | Tanpa augmentasi (debugging/dataset bersih)               | 0.0     | 0.0     | 0.0     |
| **Light**  | Augmentasi ringan untuk dataset kecil/bersih              | 0.5     | 0.0     | 0.5     |
| **Medium** | Default Ultralytics — cocok untuk kebanyakan kasus        | default | default | default |
| **Heavy**  | Agresif untuk model besar atau mencegah overfitting       | 1.0     | 0.3     | 0.5     |
| **Custom** | Konfigurasi setiap parameter augmentasi secara individual | -       | -       | -       |

### YOLO26 Model-Size Aware Defaults

Untuk model YOLO26, wizard otomatis menerapkan hyperparameter yang sudah dioptimalkan berdasarkan ukuran model:

| Size                   | lr0     | lrf    | weight_decay | momentum |
| ---------------------- | ------- | ------ | ------------ | -------- |
| nano (n)               | 0.0054  | 0.0495 | 0.00064      | 0.947    |
| small-xlarge (s/m/l/x) | 0.00038 | 0.882  | 0.00027      | 0.948    |

### Advanced Configuration

Saat memilih "Configure advanced options", tersedia kategori:

#### 1. Optimizer & Learning Rate

- **Optimizer**: auto, SGD, Adam, AdamW, NAdam, RAdam
- **Cosine LR Scheduler**: Menurunkan learning rate secara gradual
- **Initial LR (lr0)**: Learning rate awal
- **Final LR ratio (lrf)**: Rasio LR akhir terhadap LR awal
- **Momentum**: Momentum optimizer
- **Weight Decay**: Regularisasi L2
- **Warmup Epochs**: Jumlah epoch warmup
- **Warmup Momentum**: Momentum selama warmup
- **Warmup Bias LR**: Learning rate bias selama warmup
- **NBS (Nominal Batch Size)**: Untuk scaling learning rate

#### 2. Augmentation (Custom)

- **HSV-H/S/V**: Variasi warna (Hue, Saturation, Value)
- **Degrees**: Rotasi random
- **Translate**: Translasi random
- **Scale**: Scaling random
- **Shear**: Shear transformation
- **Perspective**: Perspektif transformation
- **FlipUD/FlipLR**: Flip vertikal/horizontal
- **Mosaic**: Menggabungkan 4 gambar menjadi 1
- **Mixup**: Mencampur 2 gambar
- **Copy-Paste**: Copy-paste augmentation
- **Erasing**: Random erasing

#### 3. Caching & Performance

- **Dataset Caching**: `False` (no cache), `ram` (cache di RAM), `disk` (cache di disk)
- **Mixed Precision (AMP)**: Training dengan FP16 untuk kecepatan
- **Multi-scale Training**: Variasi image size selama training
- **Rectangular Training**: Padding minimal untuk batch yang lebih efisien

#### 4. Reproducibility

- **Random Seed**: Seed untuk reproducibility (default: 0)
- **Deterministic Mode**: Memastikan hasil yang sama setiap run

#### 5. Additional Options

- **Freeze Layers**: Jumlah layer yang di-freeze (untuk fine-tuning)
- **Dropout**: Regularisasi dropout
- **Label Smoothing**: Mengurangi overconfidence pada prediksi

### Export Formats

| Format          | Deskripsi                   | Use Case                         |
| --------------- | --------------------------- | -------------------------------- |
| **ONNX**        | Cross-platform, recommended | Deployment umum                  |
| **TensorRT**    | NVIDIA GPU inference        | Inference tercepat di GPU NVIDIA |
| **CoreML**      | Apple devices               | iOS/macOS apps                   |
| **TFLite**      | Mobile/edge devices         | Android, Raspberry Pi            |
| **OpenVINO**    | Intel hardware              | Intel CPU/GPU/VPU                |
| **NCNN**        | Mobile, lightweight         | Mobile deployment ringan         |
| **TorchScript** | PyTorch deployment          | PyTorch ecosystem                |

Opsi export tambahan:

- **FP16 (Half Precision)**: Mengurangi ukuran model ~50%
- **Dynamic Input Shapes**: Mendukung berbagai ukuran input
- **INT8 Quantization**: Kompresi lebih lanjut (untuk TensorRT/TFLite)

### Smart Defaults (Otomatis)

Wizard secara otomatis menerapkan:

- `amp: true` — Mixed precision training
- `plots: true` — Generate training plots
- `val: true` — Validasi setiap epoch
- `close_mosaic: 10` — Matikan mosaic di 10 epoch terakhir
- `save_period` — Auto-save checkpoint untuk training > 100 epoch
- YOLO26 optimized hyperparameters berdasarkan model size

### Batch Size

Wizard mendukung beberapa format batch size:

- `auto` — Otomatis deteksi batch optimal (memerlukan GPU)
- `auto-70` — Gunakan 70% GPU memory
- Integer (misal `16`, `32`) — Batch size tetap

> **Catatan:** Auto-batch memerlukan GPU. Jika menggunakan CPU, wizard otomatis fallback ke batch=16.

### Menyimpan & Memuat Konfigurasi

Setelah konfigurasi training, wizard menawarkan opsi untuk menyimpan konfigurasi sebagai file YAML. File ini bisa dimuat kembali di run berikutnya untuk mengulangi training dengan parameter yang sama.

```
config-20240101_120000.yaml
```

## Struktur Output

```
runs/
└── detect/
    └── train-<project>/
        └── <run_name>/
            ├── weights/
            │   ├── best.pt          # Model terbaik
            │   └── last.pt          # Checkpoint terakhir
            ├── results.csv          # Metrics per epoch
            ├── args.yaml            # Training arguments
            ├── confusion_matrix.png
            ├── results.png
            └── ...
```

## Workflow

```
┌─────────────────────────────────────────────┐
│           YOLO Training Wizard              │
├─────────────────────────────────────────────┤
│ 1. Download Dataset (Roboflow)              │
│ 2. Configure Training (Presets/Custom)      │
│ 3. Pre-Training Health Checks               │
│ 4. Train Model (+ TensorBoard)             │
│ 5. Validate Model                           │
│ 6. Training Summary & Recommendations       │
│ 7. Export Model (ONNX/TensorRT/etc.)       │
│ 8. Zip Results                              │
└─────────────────────────────────────────────┘
```

## Tips

- **Dataset kecil (<100 images)**: Gunakan preset "Fine-Tune" dengan augmentasi "Heavy"
- **Objek kecil**: Gunakan preset "Small Objects" (imgsz=1280)
- **Training cepat untuk testing**: Gunakan preset "Quick Test"
- **Resume training**: Wizard otomatis mendeteksi training yang terinterupsi dan menawarkan resume
- **TensorBoard**: Wizard menawarkan launch TensorBoard untuk monitoring real-time

## License

MIT
