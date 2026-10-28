"""
Phát hiện hành vi VỨT RÁC bằng state machine (không phụ thuộc YOLO/OpenCV).

Mỗi vật rác (theo ID tracker) có một máy trạng thái riêng:

    IDLE ──(có chủ ở sát)──▶ HOLDING ──(tách khỏi chủ, đã từng được mang đi)──▶ SEPARATED
     ▲                        │  ▲                                                │  │
     │                        │  └────────────(chủ quay lại nhặt)─────────────────┘  │
     └──(không phải mang đi / rác bị đá, người khác cầm...)── reset                 │
                                                                                     ▼
     rơi vào thùng/xe rác ────────────────▶ DISPOSED (hợp lệ)     rác nằm yên + chủ đi xa đủ lâu
                                                                                     ▼
                                                                          ALERTED (báo 1 lần)

"Chủ" là người hoặc xe. Khoảng cách đo từ rác tới MÉP box chủ,
chia cho kích thước chủ, nên luật đúng với cả người lẫn xe, gần hay xa camera.
Thời gian tính bằng GIÂY (không phụ thuộc FPS).

Đầu vào mỗi frame: list (track_id, class_id, cx, cy, w, h).
"""
import math
from typing import Optional

from config import BehaviorConfig, ClassMap
from states import State, TrashState


def _to_box(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _size(box):
    return max(box[2] - box[0], box[3] - box[1], 1.0)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _point_box_dist(p, box):
    """Khoảng cách từ điểm tới box (0 nếu nằm trong box)."""
    dx = max(box[0] - p[0], 0.0, p[0] - box[2])
    dy = max(box[1] - p[1], 0.0, p[1] - box[3])
    return math.hypot(dx, dy)


def _inside(p, box, margin):
    return (box[0] - margin <= p[0] <= box[2] + margin and
            box[1] - margin <= p[1] <= box[3] + margin)


class LitterDetector:
    def __init__(self, cfg: BehaviorConfig = None, classes: ClassMap = None):
        self.cfg = cfg or BehaviorConfig()
        self.classes = classes or ClassMap()
        self.trash: dict[int, TrashState] = {}
        self.last_agent: dict[int, tuple] = {}      # id -> (t, box, class_id)
        self.last_container: dict[int, tuple] = {}  # id -> (t, box)
        self.t0: Optional[float] = None

    # ------------------------------------------------------------------ API chính
    def update(self, t: float, tracks):
        """
        t      : thời điểm của frame (giây)
        tracks : list (track_id, class_id, cx, cy, w, h)
        return : list sự kiện cảnh báo (dict) phát sinh ở frame này
        """
        if self.t0 is None:
            self.t0 = t
        events = []
        cm = self.classes

        agents, trashes = {}, {}
        for tid, cls, cx, cy, w, h in tracks:
            if cls in cm.agents:
                box = _to_box(cx, cy, w, h)
                agents[tid] = box
                self.last_agent[tid] = (t, box, cls)
            elif cls in cm.containers:
                self.last_container[tid] = (t, _to_box(cx, cy, w, h))
            elif cls in cm.trash:
                trashes[tid] = (cx, cy)

        for tid, pos in trashes.items():
            is_new = tid not in self.trash
            s = self.trash.setdefault(tid, TrashState())
            s.pos, s.last_seen = pos, t
            ev = self._step(t, tid, s, pos, is_new, agents)
            if ev:
                events.append(ev)

        # dọn trạng thái của rác đã mất dấu lâu
        for tid in [k for k, s in self.trash.items() if t - s.last_seen > self.cfg.trash_lost_time]:
            del self.trash[tid]
        return events

    # ------------------------------------------------------------------ state machine
    def _step(self, t, tid, s: TrashState, pos, is_new, agents):
        c = self.cfg

        if s.state in (State.ALERTED, State.DISPOSED):
            return None

        if s.state == State.IDLE:
            aid, d, ref = self._nearest_agent(pos, agents)
            if aid is not None and d < c.near_thr:
                appeared_from_hand = (is_new and c.appear_near_counts_as_carried
                                      and t - self.t0 > c.warmup_time)
                self._start_holding(t, s, aid, pos, ref, carried=appeared_from_hand)
            return None

        d, ref = self._owner_dist(t, s, pos)

        if s.state == State.HOLDING:
            if d < c.near_thr:
                # còn ở sát chủ: kiểm tra xem rác có di chuyển theo chủ không
                if _dist(pos, s.carry_origin) / ref > c.carry_move_thr:
                    s.carried = True
            elif s.carried and t - s.hold_start >= c.min_hold_time:
                # rác tách khỏi chủ sau khi đã được mang đi
                if self._in_container(t, pos):
                    s.state = State.DISPOSED          # bỏ vào thùng/xe rác: hợp lệ
                else:
                    s.state, s.sep_start, s.drop_pos = State.SEPARATED, t, pos
            else:
                # chỉ là chủ đi ngang qua rác có sẵn => bỏ qua
                self._reset(s)
            return None

        if s.state == State.SEPARATED:
            if self._in_container(t, pos):
                s.state = State.DISPOSED
                return None
            if d < c.near_thr:
                # chủ quay lại (có thể nhặt lên) => trở về HOLDING
                self._start_holding(t, s, s.owner, pos, ref, carried=True)
                return None
            if _dist(pos, s.drop_pos) / ref > c.still_thr:
                # rác không nằm yên (bị đá, người khác cầm đi, nhận nhầm...) => bỏ
                self._reset(s)
                return None
            if d > c.leave_thr and t - s.sep_start >= c.confirm_time:
                s.state = State.ALERTED
                _, _, cls = self.last_agent[s.owner]
                return {"event": "littering", "time": round(t, 2),
                        "owner_id": s.owner,
                        "owner_class": self.classes.names.get(cls, str(cls)),
                        "trash_id": tid,
                        "position": tuple(round(v) for v in pos)}
        return None

    # ------------------------------------------------------------------ hàm phụ
    def _start_holding(self, t, s, aid, pos, ref, carried):
        s.state, s.owner, s.hold_start = State.HOLDING, aid, t
        s.carry_origin, s.carried, s.ref = pos, carried, ref

    def _reset(self, s):
        s.state, s.owner, s.carried = State.IDLE, None, False

    def _nearest_agent(self, pos, agents):
        best = (None, math.inf, 100.0)
        for aid, box in agents.items():
            ref = _size(box)
            d = _point_box_dist(pos, box) / ref
            if d < best[1]:
                best = (aid, d, ref)
        return best

    def _owner_dist(self, t, s, pos):
        """Khoảng cách chuẩn hoá từ rác tới chủ; chủ mất dấu lâu => vô cùng (đã đi)."""
        info = self.last_agent.get(s.owner)
        if info is None or t - info[0] > self.cfg.owner_lost_time:
            return math.inf, s.ref
        box = info[1]
        s.ref = _size(box)
        return _point_box_dist(pos, box) / s.ref, s.ref

    def _in_container(self, t, pos):
        """Rác nằm trong (hoặc sát) box của thùng rác / xe rác, kể cả vừa mới mất dấu."""
        for ct, box in self.last_container.values():
            if t - ct <= self.cfg.container_lost_time and _inside(pos, box, self.cfg.bin_margin * _size(box)):
                return True
        return False
