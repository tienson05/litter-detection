"""
Lớp bọc model YOLO + ByteTrack.

Phần còn lại của chương trình chỉ cần `detector.track(source)` trả về từng frame
dưới dạng (frame_bgr_đã_vẽ_box, tracks), với
    tracks = list (track_id, class_id, cx, cy, w, h)

Thay đổi model: sửa `weights` và ClassMap trong config.py (nếu thứ tự lớp khác),
hoặc truyền --weights cho main.py. Nếu model không dùng ultralytics, viết lớp mới
có cùng hàm `track()` rồi trả về nó trong `build_detector()`.
"""
from typing import Iterator, Tuple

from config import ModelConfig


class TrackedDetector:
    """YOLO (ultralytics) + ByteTrack."""

    def __init__(self, cfg: ModelConfig):
        from ultralytics import YOLO  # import trễ để các module khác dùng được khi chưa cài ultralytics

        self.cfg = cfg
        self.model = YOLO(cfg.weights)

    def track(self, source) -> Iterator[Tuple[object, list]]:
        results = self.model.track(
            source=source, tracker=self.cfg.tracker, persist=True, stream=True,
            classes=self.cfg.classes.all_ids, conf=self.cfg.conf, imgsz=self.cfg.imgsz,
            device=self.cfg.device, verbose=False)
        for r in results:
            tracks = []
            if r.boxes.id is not None:
                for (cx, cy, w, h), tid, cls in zip(r.boxes.xywh.tolist(),
                                                    r.boxes.id.int().tolist(),
                                                    r.boxes.cls.int().tolist()):
                    tracks.append((tid, cls, cx, cy, w, h))
            yield r.plot(), tracks


def build_detector(cfg: ModelConfig) -> TrackedDetector:
    return TrackedDetector(cfg)
