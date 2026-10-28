"""Vẽ trạng thái rác / cảnh báo lên frame."""
import cv2

from states import State

COLORS = {State.IDLE: (200, 200, 200), State.HOLDING: (0, 200, 255),
          State.SEPARATED: (0, 120, 255), State.ALERTED: (0, 0, 255),
          State.DISPOSED: (0, 180, 0)}


def draw_states(frame, behavior, t):
    for tid, s in behavior.trash.items():
        if s.last_seen != t:
            continue
        x, y = map(int, s.pos)
        label = f"#{tid} {s.state.value}" + (f" <- {s.owner}" if s.owner is not None else "")
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[s.state], 2)
        if s.state == State.ALERTED:
            cv2.putText(frame, "VUT RAC!", (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    return frame
