"""Ghép model + state machine + vẽ + ghi video."""
import cv2

from behavior import LitterDetector
from config import BehaviorConfig, ModelConfig
from model import build_detector
from visualize import draw_states


def run_video(source, model_cfg: ModelConfig = None, behavior_cfg: BehaviorConfig = None,
              out_path="output.mp4"):
    model_cfg = model_cfg or ModelConfig()
    detector = build_detector(model_cfg)
    behavior = LitterDetector(behavior_cfg, model_cfg.classes)

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    writer = None
    for i, (frame, tracks) in enumerate(detector.track(source)):
        t = i / fps  # thời gian theo frame của video
        for ev in behavior.update(t, tracks):
            print("CANH BAO:", ev)

        frame = draw_states(frame, behavior, t)
        if writer is None:
            hh, ww = frame.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ww, hh))
        writer.write(frame)

    if writer:
        writer.release()
    print("Da luu video ket qua:", out_path)
