# Local verification

The release was checked with PyTorch using five RTTS classes and 640×640 input.
All model YAML files constructed successfully. The complete FGS-YOLO model
completed a forward pass and returned predictions with shape `[1, 9, 8400]`.

Fused-model complexity values:

| Configuration | Params (M) | GFLOPs |
|---|---:|---:|
| YOLO11n | 2.58 | 6.3 |
| SPPF-LSKA | 2.86 | 6.5 |
| GFPN | 3.19 | 7.3 |
| GFPN-Dy | 3.19 | 7.3 |
| RepGFPN | 3.66 | 8.2 |
| GDFPN | 3.67 | 8.2 |
| FGS-YOLO | 3.94 | 8.4 |

These values match the corresponding complexity entries in the manuscript.
The extracted RTTS split lists contain exactly 3,025 training, 864 validation
and 433 test images.

The default custom loss was also compared numerically with the supplied
experimental implementation on sample boxes; the maximum absolute difference
was 0.0.
