"""Train FGS-YOLO or one of its architecture ablations."""

import argparse
import os
from pathlib import Path


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(root / "configs/models/FGS-YOLO.yaml"))
    parser.add_argument("--data", default=str(root / "configs/datasets/RTTS.yaml"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="fgs_yolo_seed0")
    parser.add_argument("--loss", choices=["ciou", "focaler_iou", "piou2", "focaler_piou2"], default="focaler_piou2")
    parser.add_argument("--d", type=float, default=0.0)
    parser.add_argument("--u", type=float, default=0.95)
    parser.add_argument("--lambda-piou", type=float, default=1.3)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["FGS_BBOX_LOSS"] = args.loss
    os.environ["FGS_FOCALER_D"] = str(args.d)
    os.environ["FGS_FOCALER_U"] = str(args.u)
    os.environ["FGS_PIOU_LAMBDA"] = str(args.lambda_piou)

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        seed=args.seed,
        deterministic=True,
        pretrained=False,
        cache=False,
        close_mosaic=0,
        project="runs/train",
        name=args.name,
    )


if __name__ == "__main__":
    main()
