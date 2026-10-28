"""Cấu hình: bản đồ lớp (ClassMap), model YOLO (ModelConfig) và ngưỡng luật hành vi (BehaviorConfig)."""
from dataclasses import dataclass, field


@dataclass
class ClassMap:
    """
    Các lớp của model YOLO tự train và vai trò của từng lớp trong luật:
      trash      : vật rác cần theo dõi
      agents     : những thứ có thể là "chủ" vứt rác (người, xe)
      containers : nơi bỏ rác hợp lệ (thùng rác, xe rác) => bỏ vào đó thì KHÔNG cảnh báo
    """
    names: dict = field(default_factory=lambda: {
        0: "trash", 1: "trash-bin", 2: "person", 3: "car",
        4: "bus", 5: "truck", 6: "garbage-truck"})
    trash: tuple = (0,)
    containers: tuple = (1, 6)
    agents: tuple = (2, 3, 4, 5)

    @property
    def all_ids(self):
        return sorted(set(self.trash) | set(self.containers) | set(self.agents))


@dataclass
class ModelConfig:
    """
    Model YOLO + tracker. Đổi `weights` khi có file .pt mới; nếu thứ tự lớp khác thì sửa ClassMap.
    """
    weights: str = "trained-models/v2/best.pt"       # TODO: đường dẫn model tự train
    tracker: str = "bytetrack.yaml"
    classes: ClassMap = field(default_factory=ClassMap)
    # conf thấp để ByteTrack nhận cả box score thấp (dùng cho bước ghép lần 2, cứu vật bị che).
    # Box dưới ngưỡng "high" của tracker chỉ dùng để nối track cũ, không tạo track mới.
    conf: float = 0.1
    imgsz: int = 640
    device: str = None                     # None = tự chọn; hoặc "cpu", "0"


@dataclass
class BehaviorConfig:
    # --- khoảng cách: tính từ rác tới MÉP box của chủ, chia cho kích thước chủ (cạnh lớn nhất của box) ---
    near_thr: float = 0.25       # rác cách mép box chủ < 0.25 => "ở sát chủ"
    leave_thr: float = 1.0       # mép box chủ cách rác > 1.0 => "chủ đã bỏ đi"
    carry_move_thr: float = 0.3  # rác di chuyển > 0.3 (theo kích thước chủ) khi ở sát chủ => "được mang theo"
    still_thr: float = 0.15      # sau khi rơi, rác lệch < 0.15 => "đứng yên"
    bin_margin: float = 0.15     # nới box thùng/xe rác thêm 0.15 (theo kích thước của nó) để tính là "bỏ vào"
    # --- thời gian (giây) ---
    min_hold_time: float = 0.5   # phải ở sát chủ ít nhất bấy lâu mới tính là "cầm"
    confirm_time: float = 2.0    # rác phải nằm yên + chủ đi xa bấy lâu mới báo
    owner_lost_time: float = 1.0 # chủ mất dấu quá lâu => coi như đã rời khỏi khung hình
    trash_lost_time: float = 3.0 # rác mất dấu quá lâu => xoá trạng thái
    container_lost_time: float = 1.0  # thùng/xe rác mất dấu: vẫn dùng vị trí cũ trong bấy lâu
    # Rác trong tay thường bị che nên YOLO không thấy; nó chỉ "xuất hiện" khi vừa rơi.
    # Bật cờ này: rác MỚI xuất hiện ngay sát một chủ được coi như vừa rơi từ chủ đó.
    appear_near_counts_as_carried: bool = True
    warmup_time: float = 1.0     # vài giây đầu mọi vật đều "mới" => chưa áp dụng luật trên
