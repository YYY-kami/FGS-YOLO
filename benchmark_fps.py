"""Measure pure model-forward FPS with batch size 1."""

import argparse
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()

    from ultralytics import YOLO

    device = torch.device(args.device)
    model = YOLO(args.weights).model.to(device).eval().fuse()
    x = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x)
        times = []
        for _ in range(args.repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - start)
    latency = float(np.mean(times))
    print(f"Latency: {latency * 1000:.3f} ms/image")
    print(f"FPS: {1.0 / latency:.2f}")


if __name__ == "__main__":
    main()
