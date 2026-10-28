# Training Folder Guide

Để dễ quản lý code và kết quả train, thống nhất cách sử dụng các thư mục như sau nhé!

## 1. Code train

Code dùng để train model được đặt trong thư mục:

`training/train/`

## 2. Kết quả báo cáo

Các kết quả sau khi train như biểu đồ, file CSV, confusion matrix, ROC curve và các metrics được lưu vào thư mục:

`training/reports/vX-DDMMYY/`

Ví dụ: `training/reports/v2-281026/`

## 3. Model weights

Các file model weights và checkpoint được lưu trong thư mục:

`trained_models/vX-DDMMYY/`

Ví dụ: `trained_models/v2-281026/`

Tên thư mục trong `reports/` và `trained_models/` cần tương ứng với nhau để dễ đối chiếu kết quả và model của từng lần train.

## 4. Cách đặt tên thư mục

Sử dụng format `vX-DDMMYY`, trong đó:
- `X`: Số phiên bản train.
- `DDMMYY`: Ngày train theo định dạng ngày-tháng-năm.

Ví dụ: Lần train tiếp theo có thể đặt tên là `v3-091126`.

## 5. Một vài lưu ý

- Mỗi lần train mới, tạo thư mục phiên bản tương ứng để lưu kết quả nhé.
- Giữ lại kết quả của các lần train trước để tiện so sánh và theo dõi tiến trình.
- Lưu file đúng thư mục để dễ tìm kiếm và sử dụng lại khi cần.

Cảm ơn nhé!