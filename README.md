# Exam Proctoring Unified

Ứng dụng giám sát thi offline hợp nhất từ ExamWatch v2 và AI Exam Control. Project mới có backend FastAPI, giao diện web tĩnh, SQLite, luồng camera/microphone, model video và model chép lời tiếng Việt.

## Chức năng đã tích hợp trong mã nguồn

- Webcam trình duyệt; dashboard hiển thị người, khung xương, tư thế, cảm xúc và trạng thái model.
- Đăng ký khuôn mặt cục bộ, đối chiếu danh tính tùy chọn trong phiên và theo dõi tạm thời bằng `track_id`.
- Phát hiện điện thoại bằng checkpoint YOLO `phone_detector_v5.pt`; cảnh báo được gắn vào người khi có pose đủ rõ, tạo clip bằng chứng và có cooldown.
- Phát hiện nhổm dậy, giơ tay, nghiêng thân và cảm xúc tiêu cực kéo dài.
- Microphone tùy chọn, chép lời tiếng Việt theo phiên; không nhận dạng người nói.
- Hiệu chuẩn tư thế ngồi, ngưỡng cảnh báo nhổm dậy, cảnh báo kèm ảnh hoặc video trước/sau sự kiện.
- Danh sách thí sinh, phiên thi, lịch sử, transcript, duyệt cảnh báo và xuất JSON.
- Lưu cục bộ trong SQLite và `data/evidence/`; khi cấu hình Firebase, dữ liệu được đồng bộ thêm lên Firestore và evidence lên Cloud Storage.

Đây là công cụ hỗ trợ giám thị. Nhãn cảm xúc/tư thế và phát hiện điện thoại cần được xem cùng bằng chứng; hệ thống không tự kết luận vi phạm.

## Chạy trên Windows

Cần Python 3.12 x64. Lần chuẩn bị đầu tiên cần mạng để cài thư viện. Các model ONNX, Pose Lite, Zipformer và YOLO được đóng gói trong `models/`.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\prepare.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

Để mở dữ liệu trình diễn có sẵn mà không cần đăng nhập Firebase, chạy với `-Demo`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1 -Demo
```

Chế độ mô phỏng dùng phiên/cảnh báo mẫu trên máy và chỉ được thiết kế cho máy chủ localhost.

Mở http://127.0.0.1:8000. Cho phép camera/microphone khi trình duyệt hỏi. Bản hiện tại dùng giao diện ExamWatch; pipeline điện thoại từ AI Exam Control đã được nối vào cùng luồng phiên, cảnh báo và bằng chứng.

## Firebase

Đọc [FIREBASE_SETUP.md](FIREBASE_SETUP.md) để bật Firestore + Storage và cấu hình service account. SQLite vẫn là nguồn ghi chính tại chỗ; đồng bộ cloud chạy nền. Trạng thái nằm trong `/api/status`.

## Model và dữ liệu

- `models/phone_detector_v5.pt`: model phát hiện điện thoại từ AI Exam Control.
- `models/yunet.onnx`, `sface.onnx`, `emotion.onnx`, `pose_lite.task`: nhận diện mặt, embedding, cảm xúc và pose từ ExamWatch.
- `models/zipformer-vi-30m/`: model chép lời tiếng Việt.
- `data/`: CSDL SQLite và tệp bằng chứng phát sinh; không commit dữ liệu phiên.

Kiểm tra giấy phép trong các dự án nguồn trước khi phân phối lại các model. Các model chỉ được giữ tại project mới để phục vụ bản hợp nhất trong workspace này.

## Cấu trúc

```text
examwatch/                 FastAPI, session, vision, tracking, audio và lưu trữ
web/                       Dashboard giám thị
models/                    Model cục bộ
scripts/                   Chuẩn bị model
packaging/windows/         Khóa phụ thuộc cho bộ cài Windows
```

## Giới hạn hiện tại

- Bản đầu nhắm đến demo cục bộ tối đa 3 người; hiệu năng phụ thuộc CPU/máy.
- Điện thoại chỉ gắn với một track khi tâm hộp điện thoại nằm trong vùng pose rõ ràng, duy nhất; trường hợp không chắc không gán người.
- Nhận diện danh tính là tính năng cục bộ dựa trên đăng ký; không dùng `track_id` làm danh tính.
- Phát hiện quay đầu dùng độ lệch mũi so với trục mắt trên landmark YuNet; cảnh báo sau 1,25 giây, tối thiểu 3 quan sát và reset khi gián đoạn trên 0,75 giây. Đây là heuristic gọn để hợp nhất, không phải phép đo yaw đã benchmark.
- Chưa hỗ trợ video file làm nguồn đầu vào trong giao diện.
