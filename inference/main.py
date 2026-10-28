"""
Điểm vào:
    python src/main.py video.mp4
    python src/main.py video.mp4 --weights weights/best.pt
Thứ tự lớp của model (trash, trash-bin, person, car, bus, truck, garbage-truck) khai báo ở ClassMap trong config.py.
"""
import argparse
from dataclasses import replace

from config import ModelConfig
from pipeline import run_video


def parse_args():
    d = ModelConfig()
    p = argparse.ArgumentParser(description="Phát hiện hành vi vứt rác")
    p.add_argument("source", help="đường dẫn video (hoặc 0 cho webcam)")
    p.add_argument("--out", default="output.mp4")
    p.add_argument("--weights", default=d.weights, help="file .pt của model YOLO")
    p.add_argument("--conf", type=float, default=d.conf)
    p.add_argument("--device", default=d.device)
    return p.parse_args()


def main():
    a = parse_args()
    source = int(a.source) if a.source.isdigit() else a.source
    print(f"Models: {a.weights}")
    cfg = replace(ModelConfig(), weights=a.weights, conf=a.conf, device=a.device)
    run_video(source, cfg, out_path=a.out)


if __name__ == "__main__":
    main()
