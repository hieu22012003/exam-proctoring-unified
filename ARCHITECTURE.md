# Kiến trúc project hợp nhất

```text
Browser webcam + microphone
        │ JPEG frames / PCM audio
        ▼
FastAPI + Runtime (một phiên cục bộ)
  ├─ YuNet + SFace ──> nhận diện mặt / đối chiếu gallery tùy chọn
  ├─ MediaPipe Pose ─> tracking, tư thế, góc tay/thân, nhổm dậy
  ├─ Emotion ONNX ───> tín hiệu cảm xúc tham khảo
  ├─ YOLO Phone ─────> phát hiện điện thoại / gắn track khi rõ ràng
  ├─ Zipformer ──────> transcript tiếng Việt (worker audio riêng)
  ├─ Evidence ring buffer ─> clip WebM trước/sau sự kiện
  └─ SQLite + evidence files
        ▲
        └──────── REST API / static dashboard
```

## Hợp nhất cụ thể

- Dùng ExamWatch làm nền cho lifecycle phiên, capture, SQLite, lưu transcript, face/identity, cảm xúc, pose và giao diện.
- Chuyển checkpoint `phone_detector_v5.pt` từ AI Exam Control sang `models/` của project mới; `Vision` chạy YOLO trong cùng frame pipeline.
- Gán điện thoại vào một track chỉ khi tâm hộp phát hiện nằm trong vùng pose của đúng một người. Sự kiện `phone` dùng cùng Store và ring buffer để tạo clip, có cooldown 6 giây.
- Ước lượng quay đầu từ vị trí mũi so với trung điểm hai mắt của YuNet. Chuỗi nghi vấn cần liên tục 1,25 giây, ít nhất 3 mẫu; gap trên 0,75 giây đặt lại bộ đếm. Cảnh báo tạo clip qua cùng ring buffer.
- Giao diện hiển thị model YOLO Phone, nhãn phát hiện điện thoại trong danh sách/ngay trên overlay và sự kiện trong lịch sử.
- Giữ việc theo dõi danh tính tách biệt với ID tạm của tracker.

## Dữ liệu và API

Backend hiện dùng API của ExamWatch: phiên, thí sinh, frame/audio ingest, cảnh báo, evidence, transcript và export. SQLite `examwatch.sqlite3` lưu các phiên, danh tính đăng ký, cảnh báo, transcript và quan sát. Clip/ảnh nằm trong `data/evidence/`.

## Giới hạn và việc còn lại

- Luật quay đầu được triển khai lại bằng tỉ lệ hình học YuNet; chưa port nguyên thuật toán yaw/pose YOLO của AI Exam Control và chưa benchmark ngưỡng.
- Input file video và dashboard React từ AI Exam Control chưa được tích hợp; project dùng dashboard tĩnh đã có webcam/micro.
- Ngưỡng YOLO phone mặc định là 0.35 và cooldown 6 giây; chưa có màn hình chỉnh ngưỡng này.
- Các model gốc có điều kiện giấy phép riêng; cần đọc thông báo tương ứng trước khi phát hành.
