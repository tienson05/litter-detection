"""Kiểu dữ liệu dùng chung: trạng thái của vật "rác" trong state machine."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class State(Enum):
    IDLE = "idle"            # rác chưa liên quan tới ai
    HOLDING = "holding"      # rác đang ở sát một chủ (có thể đang được cầm)
    SEPARATED = "separated"  # rác vừa tách khỏi chủ, đang chờ xác nhận
    ALERTED = "alerted"      # đã cảnh báo vứt rác, không báo lại nữa
    DISPOSED = "disposed"    # rác đã bỏ vào thùng/xe rác => hợp lệ, không cảnh báo


@dataclass
class TrashState:
    state: State = State.IDLE
    owner: Optional[int] = None   # ID chủ (người hoặc xe)
    hold_start: float = 0.0
    carry_origin: tuple = (0.0, 0.0)
    carried: bool = False         # đã thấy rác di chuyển cùng chủ chưa
    sep_start: float = 0.0
    drop_pos: tuple = (0.0, 0.0)
    ref: float = 100.0            # kích thước chủ gần nhất (để chuẩn hoá)
    pos: tuple = (0.0, 0.0)
    last_seen: float = 0.0
