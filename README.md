# FGS-YOLO

Official research code prepared for the manuscript **“FGS-YOLO: A Multi-Stage
Feature and Localization Optimization Network for Traffic Object Detection in
Foggy Scenes.”**

This release is based on Ultralytics 8.3.9 and contains only code and split
metadata. RTTS images, trained weights, caches and experiment outputs are not
redistributed.

## Method components

- **SPPF-LSKA** for deep multi-scale context aggregation and large-kernel spatial modeling.
- **GDFPN** with multi-path fusion, CSPStage and LP-mode DySample upsampling.
- **Focaler-PIoU2** for bounding-box regression, with default `d=0`, `u=0.95`, `lambda=1.3`.

The model configuration is `configs/models/FGS-YOLO.yaml`. The implementation
of the four custom network classes is isolated in
`ultralytics/nn/fgs_modules.py`; the regression implementation is in
`ultralytics/utils/fgs_loss.py`.

## Environment

The paper used Python 3.10.8, PyTorch 2.1.2, CUDA 12.1 and an NVIDIA RTX 3090.

```bash
conda create -n fgs-yolo python=3.10.8 -y
conda activate fgs-yolo
# Install the PyTorch 2.1.2 build appropriate for your CUDA environment first.
pip install -r requirements-fgs.txt
```

## RTTS preparation

Download RTTS from the official dataset source and create this layout:

```text
datasets/RTTS_split/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

The manuscript uses a fixed 7:2:1 split of 3,025/864/433 images. File lists are
provided in `splits/RTTS/`. No image is included in this repository.

## Train the complete model

```bash
python train.py --device 0 --seed 0 --name fgs_yolo_seed0
```

The defaults reproduce the manuscript protocol: 640×640 input, 200 epochs,
batch size 16, SGD, initial/final learning rates 0.01/0.001, momentum 0.937,
weight decay 0.0005, seed 0, no pretrained weights and Focaler-PIoU2.

Random-seed runs:

```bash
python train.py --seed 0    --name fgs_yolo_seed0
python train.py --seed 42   --name fgs_yolo_seed42
python train.py --seed 3407 --name fgs_yolo_seed3407
```

## Evaluate on the test set

```bash
python val.py --weights runs/train/fgs_yolo_seed0/weights/best.pt --batch 1 --split test
```

The script reports P, R, mAP@0.5, mAP@0.75, mAP@0.5:0.95, GFLOPs,
parameters and inference-only FPS, then writes `paper_metrics.txt` under the
validation run directory.

For a dedicated forward-pass speed test:

```bash
python benchmark_fps.py --weights runs/train/fgs_yolo_seed0/weights/best.pt --device cuda:0
```

## Ablations

Architecture configurations included in `configs/models/`:

- `yolo11.yaml`: YOLO11 baseline.
- `yolo11-SPPF-LSKA.yaml`: SPPF-LSKA ablation.
- `yolo11-GFPN.yaml`: multi-path GFPN with C3k2 and fixed upsampling.
- `yolo11-GFPN-Dy.yaml`: GFPN plus DySample.
- `yolo11-RepGFPN.yaml`: CSPStage/RepConv aggregation with fixed upsampling.
- `yolo11-GDFPN.yaml`: CSPStage/RepConv aggregation plus DySample.
- `FGS-YOLO.yaml`: complete architecture.

Use `--loss ciou` for the YOLO11n, SPPF-LSKA, GFPN and GDFPN architecture-only
ablations. Use `--loss focaler_piou2` for the complete FGS-YOLO configuration.

Loss ablations use the same architecture command with one of:

```bash
python train.py --loss ciou
python train.py --loss focaler_iou
python train.py --loss piou2
python train.py --loss focaler_piou2 --d 0 --u 0.95 --lambda-piou 1.3
```

Sensitivity examples:

```bash
python train.py --loss focaler_piou2 --d 0.05 --u 0.95 --lambda-piou 1.3
python train.py --loss focaler_piou2 --d 0    --u 0.90 --lambda-piou 1.3
python train.py --loss focaler_piou2 --d 0    --u 0.95 --lambda-piou 1.4
```

## Reproducibility note

Read `docs/MANUSCRIPT_CODE_CONSISTENCY.md` before public release. It records one
important difference between the loss equation currently written in the
manuscript and the implementation that produced the reported results. Resolve
that difference explicitly before publication.

## Citation and license

Replace the placeholders in `CITATION.cff.template`, rename it to
`CITATION.cff`, and add the final paper DOI after acceptance.

This repository retains the Ultralytics GNU AGPL-3.0 license. See `LICENSE` and
`THIRD_PARTY_NOTICES.md`. If you distribute a modified network service, review
the AGPL obligations and preserve the licenses/notices of all adapted sources.
