"""PyTorch-free YOLO detector on ONNX Runtime (for small hosts).

Mirrors Ultralytics' predict pipeline for a single-output detect model: letterbox to
the export size, decode [1, 4 + classes, anchors], class-agnostic NMS, scale back."""
from __future__ import annotations

import cv2
import numpy as np


class OnnxYolo:
    def __init__(self, path, threads=1):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2])

    def letterbox(self, frame):
        h, w = frame.shape[:2]
        r = min(self.size / h, self.size / w)
        nw, nh = round(w * r), round(h * r)
        pad_w, pad_h = (self.size - nw) / 2, (self.size - nh) / 2
        image = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else frame
        top, bottom = round(pad_h - .1), round(pad_h + .1)
        left, right = round(pad_w - .1), round(pad_w + .1)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return image, r, left, top

    def predict(self, frame, conf=0.25, iou=0.7, max_det=50):
        """Boxes as (x1, y1, x2, y2, confidence, class) in frame pixels."""
        image, r, left, top = self.letterbox(frame)
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)
        out = self.session.run(None, {self.input_name: blob})[0][0]  # (4 + nc, anchors)
        scores = out[4:]
        cls = scores.argmax(0)
        best = scores[cls, np.arange(scores.shape[1])]
        keep = best >= conf
        if not keep.any():
            return []
        cx, cy, bw, bh = out[:4, keep]
        best, cls = best[keep], cls[keep]
        x1, y1 = (cx - bw / 2 - left) / r, (cy - bh / 2 - top) / r
        x2, y2 = (cx + bw / 2 - left) / r, (cy + bh / 2 - top) / r
        h, w = frame.shape[:2]
        x1, x2 = np.clip(x1, 0, w), np.clip(x2, 0, w)
        y1, y2 = np.clip(y1, 0, h), np.clip(y2, 0, h)
        rects = [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in zip(x1, y1, x2, y2)]
        picked = cv2.dnn.NMSBoxes(rects, best.astype(float).tolist(), conf, iou)
        picked = np.array(picked).reshape(-1)[:max_det]
        order = picked[np.argsort(-best[picked])]
        return [(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), float(best[i]), int(cls[i])) for i in order]
