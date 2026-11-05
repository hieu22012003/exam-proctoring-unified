# Model runtime

Mặc định dùng Zipformer 30M nguyên mẫu tiếng Việt của hynt, qua Sherpa-ONNX
INT8 trên CPU. Chạy `prepare.ps1` để tải YuNet, SFace, Emotion ONNX, Pose Lite
và Zipformer. Không cần PyTorch/CUDA để chạy demo.
`scripts/prepare_zipformer.py` kiểm tra SHA-256 của bản phát hành cố định.
Trọng số không được commit vào Git. Các checkpoint Whisper/LoRA là tùy chọn
thử nghiệm, cần chuẩn bị riêng; xem ZIPFORMER_WEB.md và tài liệu training.

`manifest.json` ghi nguồn, kích thước và SHA-256 của các file được chuẩn bị.
Ứng dụng chỉ đọc đường dẫn local; không tự tải model trong phiên thi.
Không đổi các file model sang kiến trúc khác mà chưa cập nhật adapter tiền xử lý.
