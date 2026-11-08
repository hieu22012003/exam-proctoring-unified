"""One-time online preparation. Normal application startup never downloads weights."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = {
    "yunet.onnx": (
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx", 100_000),
    "sface.onnx": (
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx", 1_000_000),
    "emotion.onnx": (
        "https://huggingface.co/serengil/deepface/resolve/main/facial_expression_model_weights.onnx", 100_000),
    "pose_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task", 1_000_000),
    # Face blendshapes (eye closure, yawning) for the "tired" support state.
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", 1_000_000),
    # Multi-person body pose for full classrooms (AGPL-3.0, Ultralytics).
    "yolo11n-pose.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt", 1_000_000),
}
LICENSES = {
    "yunet-LICENSE.txt": "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/LICENSE",
    "sface-LICENSE.txt": "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_recognition_sface/LICENSE",
    "whisper-MIT.txt": "https://raw.githubusercontent.com/openai/whisper/main/LICENSE",
}


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def download(folder, item):
    name, (url, min_bytes) = item
    target = folder / name
    if not target.exists() or target.stat().st_size < min_bytes:
        print(f"Downloading {name}...", flush=True)
        temporary = target.with_suffix(".partial")
        request = urllib.request.Request(url, headers={"User-Agent": "ExamWatch-Local/1.0"})
        for attempt in range(6):  # model hosts sometimes answer 5xx to build machines
            try:
                with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as dest:
                    while chunk := source.read(1024 * 1024):
                        dest.write(chunk)
                if temporary.stat().st_size >= min_bytes:
                    break
                error = RuntimeError(f"Invalid download: {name}")
            except Exception as exc:
                error = exc
            if attempt == 5:
                raise RuntimeError(f"Could not download {name}: {error}")
            wait = min(60, 5 * 2 ** attempt)
            print(f"{name}: attempt {attempt + 1} failed ({error}); retrying in {wait}s", flush=True)
            time.sleep(wait)
        temporary.replace(target)
    return {"file": name, "url": url, "bytes": target.stat().st_size, "sha256": digest(target)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper", choices=["tiny", "base", "phowhisper-base", "zipformer-vi-30m"], default="zipformer-vi-30m")
    args = parser.parse_args()
    folder = Path(os.getenv("EXAMWATCH_MODELS", str(ROOT / "models"))).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(ROOT / ".tmp" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda item: download(folder, item), ARTIFACTS.items()))
    license_folder = ROOT / "licenses"
    license_folder.mkdir(exist_ok=True)
    for name, url in LICENSES.items():
        target = license_folder / name
        if not target.is_file():
            with urllib.request.urlopen(url, timeout=30) as source:
                target.write_bytes(source.read())
    if args.whisper in {"phowhisper-base","zipformer-vi-30m"}:
        if args.whisper == 'zipformer-vi-30m':
            sys.path.insert(0,str(ROOT))
            from scripts.prepare_zipformer import main as prepare_zipformer
            prepare_zipformer()
        elif not (folder / "whisper-phowhisper-base/model.bin").is_file():
            raise SystemExit("Copy models/whisper-phowhisper-base from the prepared demo, or run prepare_phowhisper.ps1 once. Runtime does not require PyTorch.")
        print(f"Using local {args.whisper} INT8. Verifying initialization...", flush=True)
        sys.path.insert(0, str(ROOT))
        from examwatch.config import Settings
        from examwatch.vision import Vision
        from examwatch.speech import Speech
        settings = Settings.load()
        settings.whisper_name = args.whisper
        vision = Vision(settings)
        vision.close()
        Speech(settings)
        print("OK. Offline inference is ready. Run start.ps1", flush=True)
        return
    from huggingface_hub import HfApi, snapshot_download
    repository = f"Systran/faster-whisper-{args.whisper}"
    print(f"Downloading {repository}...", flush=True)
    revision = HfApi().model_info(repository).sha
    snapshot_download(repository, revision=revision, local_dir=str(folder / f"whisper-{args.whisper}"),
                      allow_patterns=["model.bin", "config.json", "tokenizer.json", "vocabulary.*", "preprocessor_config.json"])
    for path in (folder / f"whisper-{args.whisper}").glob("*"):
        if path.is_file():
            records.append({"file": path.relative_to(folder).as_posix(), "repository": repository,
                            "revision": revision, "bytes": path.stat().st_size, "sha256": digest(path)})
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_files = {r["file"] for r in records}
        records.extend(r for r in old.get("artifacts", []) if r["file"] not in current_files
                       and (folder / r["file"]).is_file())
    manifest = {"prepared_at": datetime.now(timezone.utc).isoformat(), "artifacts": records}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("All models are local. Verifying initialization...", flush=True)
    sys.path.insert(0, str(ROOT))
    from examwatch.config import Settings
    from examwatch.vision import Vision
    from examwatch.speech import Speech
    settings = Settings.load()
    settings.whisper_name = args.whisper
    vision = Vision(settings)
    vision.close()
    Speech(settings)
    print("OK. Offline inference is ready. Run start.ps1", flush=True)


if __name__ == "__main__":
    main()
