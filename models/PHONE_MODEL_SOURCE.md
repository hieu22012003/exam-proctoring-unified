# 🎓 AI Exam Proctoring Engine (V7 Multi-Angle Robust)

Module AI giám sát thi tự động tích hợp phát hiện điện thoại và tư thế nghi vấn gian lận đa góc nhìn.

---

## 📦 Các thành phần trong thư mục này

1. **`weights/`**:
   - `phone_detector_v5.pt` (~19.2 MB): Model YOLO phát hiện điện thoại (train từ `runs/detect/runs/phone_detector_v5`).
   - `yolo11m-pose.pt` (~42.5 MB): Model YOLO Pose estimation 17 keypoints theo chuẩn COCO.
2. **`exam_analyzer.py`**:
   - Chứa Class `ExamBehaviorDetector` (hoặc alias `ExamMonitorEngine`).
   - Giữ nguyên 100% thuật toán V7:
     - Lọc tọa độ qua thời gian (EMA `KeypointSmoother` được thiết kế nhằm giảm rung giật pixel).
     - Tách biệt góc nghiêng đầu (Yaw vs Roll) và che khuất tai.
     - Phát hiện quay đầu sang bài bạn, cúi gập người xoay ra sau (`TurnBack`), lệch trục vai (`Lateral Drift`), vươn tay xa (`Wrist Reach`).
     - Gán điện thoại vào cổ tay/thí sinh gần nhất (`assign_phones_to_persons`).
     - Bộ đếm thời gian nghi vấn nhằm hạn chế báo động tức thời (`ALERT_SECONDS = 1.25s`).

---

## 🚀 Khởi chạy Máy chủ Backend Chính thức

Hệ thống sử dụng duy nhất một entrypoint backend sản xuất:

```bash
# Khởi chạy FastAPI backend engine tại cổng 8000:
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
*(Ghi chú: Toàn bộ script độc lập cũ như `test_demo.py` và `api_server.py` đã được lưu trữ sang `archive/legacy/` nhằm tránh nhầm lẫn với entrypoint chính).*

