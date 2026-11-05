"""Bounded JPEG pre-roll and background VP8 evidence encoding (no audio)."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading
import cv2
import numpy as np


class EvidenceClips:
    def __init__(self, directory):
        self.directory = directory
        self.frames = deque(maxlen=85)
        self.pending = {}
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='evidence-video')
        self.closed = False
        self.slots = threading.BoundedSemaphore(6)

    def push(self, frame, now):
        with self.lock:
            if self.closed or (self.frames and now-self.frames[-1][0] < .2):
                return
            h, w = frame.shape[:2]
            scale = min(640/w, 480/h)
            size = (max(2,round(w*scale/2)*2), max(2,round(h*scale/2)*2))
            small = cv2.resize(frame, size)
            ok, jpeg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                return
            item = (now, jpeg.tobytes())
            self.frames.append(item)
            while self.frames and now-self.frames[0][0] > 10:
                self.frames.popleft()
            for job in self.pending.values():
                if now <= job['end']:
                    job['frames'].append(item)

    def trigger(self, filename, now, caption):
        with self.lock:
            if self.closed or not self.slots.acquire(blocking=False):
                (self.directory / filename).with_suffix('.failed').write_text(
                    'Video encoder busy or closed',encoding='utf-8')
                return
            timer = threading.Timer(5, self.finish, args=(filename,))
            timer.daemon = True
            self.pending[filename] = dict(frames=list(self.frames), end=now+5,
                                          timer=timer, caption=caption)
            timer.start()

    def finish(self, filename):
        with self.lock:
            job = self.pending.pop(filename, None)
            if job:
                job['timer'].cancel()
                self.pool.submit(self.encode, filename, job)

    def encode(self, filename, job):
        target = self.directory / filename
        temporary = target.with_name(target.stem+'.writing.webm')
        writer = None
        try:
            frames = job['frames']
            if not frames:
                raise ValueError('No video frames')
            first = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
            height, width = first.shape[:2]
            writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'VP80'), 5, (width,height))
            if not writer.isOpened():
                raise OSError('VP8 encoder unavailable')
            # Render on an actual time grid; repeated frames do not imply new observations.
            start, end = frames[0][0], frames[-1][0]
            index = 0
            for step in range(max(1, round((end-start)*5)+1)):
                stamp = start + step/5
                while index+1 < len(frames) and frames[index+1][0] <= stamp:
                    index += 1
                image = cv2.imdecode(np.frombuffer(frames[index][1],np.uint8),cv2.IMREAD_COLOR)
                image = cv2.resize(image,(width,height))
                cv2.putText(image, job['caption'], (10,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,80,255),1)
                cv2.putText(image, f'+{stamp-start:.1f}s | source age {stamp-frames[index][0]:.1f}s',
                            (10,height-12),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,80,255),1)
                writer.write(image)
            writer.release()
            writer = None
            if temporary.stat().st_size < 100:
                raise OSError('Empty evidence video')
            temporary.replace(target)
        except Exception as exc:
            target.with_suffix('.failed').write_text(str(exc),encoding='utf-8')
        finally:
            if writer is not None:
                writer.release()
            temporary.unlink(missing_ok=True)
            self.slots.release()

    def reset(self):
        with self.lock:
            for name in list(self.pending):
                self.finish(name)
            self.frames.clear()

    def close(self):
        with self.lock:
            self.reset()
            self.closed = True
        self.pool.shutdown(wait=True)
