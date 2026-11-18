"""Evaluate a checkpoint on the RTTS test split and save paper-style metrics."""

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default=str(root / "configs/datasets/RTTS.yaml"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="fgs_yolo_test")
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import model_info

    model = YOLO(args.weights)
    result = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/val",
        name=args.name,
    )
    _, parameters, _, gflops = model_info(model.model)
    p = result.results_dict["metrics/precision(B)"] * 100
    r = result.results_dict["metrics/recall(B)"] * 100
    map50 = result.results_dict["metrics/mAP50(B)"] * 100
    map75 = float(np.mean(result.box.all_ap[:, 5])) * 100
    map5095 = result.results_dict["metrics/mAP50-95(B)"] * 100
    inference_ms = result.speed["inference"]
    fps = 1000.0 / inference_ms

    text = (
        "FGS-YOLO evaluation\n"
        f"split={args.split}, imgsz={args.imgsz}, batch={args.batch}\n"
        f"P(%)={p:.2f}\nR(%)={r:.2f}\nmAP@0.5(%)={map50:.2f}\n"
        f"mAP@0.75(%)={map75:.2f}\nmAP@0.5:0.95(%)={map5095:.2f}\n"
        f"GFLOPs={gflops:.1f}\nParams(M)={parameters / 1e6:.2f}\n"
        f"Inference(ms/image)={inference_ms:.3f}\nFPS(inference only)={fps:.2f}\n"
    )
    output = Path(result.save_dir) / "paper_metrics.txt"
    output.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
