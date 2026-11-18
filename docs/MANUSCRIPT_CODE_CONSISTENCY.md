# Manuscript–code consistency audit

## Consistent items

- Baseline: YOLO11n from Ultralytics 8.3.9.
- Deep feature module: the original SPPF is replaced by SPPF-LSKA; LSKA uses
  separable depthwise kernels with effective kernel size 11.
- Neck: GDFPN contains multi-path fusion, CSPStage aggregation and two DySample
  upsampling nodes in LP mode.
- Detection output: P3, P4 and P5 three-scale outputs are retained.
- Regression defaults: Focaler-PIoU2 with d=0, u=0.95 and lambda=1.3; DFL remains enabled.
- RTTS split: 3,025/864/433 images for train/validation/test.
- Main settings exposed by `train.py`: 640 pixels, 200 epochs, batch 16, SGD,
  initial/final learning rates 0.01/0.0001, momentum 0.937, weight decay
  0.0005, seed 0, no pretrained weights and mosaic enabled throughout training.
- `val.py` defaults to the test split and batch size 1, matching the manuscript's
  reported test protocol and FPS setting.
- Local construction with five classes reproduces the manuscript complexity:
  3.94 M fused parameters and 8.4 GFLOPs for FGS-YOLO. See `VERIFICATION.md`.

## Important formula discrepancy to resolve before publication

The experimental source computes PIoU v2 after replacing raw IoU with the
Focaler-remapped IoU. Therefore the implemented loss is equivalent to applying
the PIoU v2 quality weight to `1 - IoU_F + f(P)`.

The current manuscript formula instead writes
`L_PIoUv2(raw IoU) + IoU - IoU_F`. These expressions are not generally equal.
The release keeps the implementation that generated the reported experiments.
Before publication, either revise the equation/description to match the actual
implementation or rerun the experiments with code changed to the manuscript
equation. Do not silently change the implementation while retaining old results.

## Repairs made relative to the supplied archive

- Replaced the missing `111.yaml` training entry with `FGS-YOLO.yaml`.
- Removed hard-coded `/root/...` paths and provided a portable dataset YAML.
- Added explicit `seed=0` and `pretrained=False`.
- Changed evaluation default from validation split to test split.
- Added configurable loss type and d/u/lambda settings for the loss and
  sensitivity ablations.
- Added separate GFPN, GFPN-Dy, RepGFPN and GDFPN YAML files so the fine-grained
  neck ablation in the manuscript can be reproduced without renaming models.
- Isolated the FGS modules from unrelated experimental modules and dependencies.
