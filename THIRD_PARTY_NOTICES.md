# Third-party components

Ứng dụng sử dụng các thư viện đóng gói; không cần giữ toàn bộ cây mã nguồn upstream.
Giấy phép dependency vẫn nằm trong distribution tương ứng ở `.venv`.

| Thành phần | Nguồn | Giấy phép / ghi chú |
| --- | --- | --- |
| DeepFace Emotion ONNX, thứ tự nhãn và tiền xử lý | https://github.com/serengil/deepface | MIT; xem `licenses/deepface-MIT.txt`. Model: https://huggingface.co/serengil/deepface |
| YuNet / SFace | https://github.com/opencv/opencv_zoo | Giữ giấy phép riêng từng model trong `licenses/` khi chuẩn bị |
| MediaPipe Pose Landmarker Lite | https://github.com/google-ai-edge/mediapipe | Apache-2.0; xem `licenses/mediapipe-Apache-2.0.txt` |
| Faster-Whisper | https://github.com/SYSTRAN/faster-whisper | MIT; xem `licenses/faster-whisper-MIT.txt` |
| Whisper tiny weights | https://huggingface.co/Systran/faster-whisper-tiny | Model chuyển sang CTranslate2 từ OpenAI Whisper; MIT |
| Zipformer 30M (mặc định) | https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h | CC-BY-NC-ND-4.0; bản nguyên mẫu dùng cho demo phi thương mại; không bao gồm weights trong Git |
| Sherpa-ONNX | https://github.com/k2-fsa/sherpa-onnx | Apache-2.0; giữ giấy phép trong distribution |
| PhoWhisper-base (tùy chọn) | https://huggingface.co/vinai/PhoWhisper-base | BSD-3-Clause; xem `licenses/phowhisper-BSD-3-Clause.txt` |

Model mặc định đang chạy là Zipformer nguyên mẫu. Các script và tài liệu thử
fine-tune được giữ để tham khảo; không đưa checkpoint thử nghiệm vào Git.
Nguồn và hash cụ thể được ghi trong manifest/source.json lúc chuẩn bị model.
