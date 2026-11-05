from math import gcd
from collections import Counter
import re
import numpy as np


def unreliable_segment(segment):
    """Reject uncertain decoding and repetition loops, without editing recognized words."""
    if (segment.no_speech_prob >= 0.6 or getattr(segment, "avg_logprob", 0) < -1.0
            or getattr(segment, "compression_ratio", 0) > 2.2):
        return True
    tokens = re.findall(r"\w+", segment.text.casefold())
    if not tokens:
        return True
    if len(tokens) >= 8:
        trigrams = Counter(tuple(tokens[i:i+3]) for i in range(len(tokens)-2))
        if max(trigrams.values(), default=0) >= 3:
            return True
        if Counter(tokens).most_common(1)[0][1] / len(tokens) > 0.65:
            return True
    return False


class Speech:
    def __init__(self, settings):
        self.zipformer = settings.whisper_name in {'zipformer-vi-30m','zipformer-vi-30m-lora','zipformer-vi-30m-lora-low-lr20'}
        if self.zipformer:
            import sherpa_onnx
            root = settings.models / settings.whisper_name
            self.model = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(root/'encoder.int8.onnx'), decoder=str(root/'decoder.onnx'),
                joiner=str(root/'joiner.int8.onnx'), tokens=str(root/'tokens.txt'),
                num_threads=settings.threads, sample_rate=16000, feature_dim=80,
                decoding_method='greedy_search', provider='cpu')
            self.last_warning = None
            return
        from faster_whisper import WhisperModel
        self.model = WhisperModel(str(settings.models / f"whisper-{settings.whisper_name}"),
                                  device="cpu", compute_type="int8", cpu_threads=settings.threads,
                                  num_workers=1, local_files_only=True)
        self.last_warning = None

    def transcribe(self, samples, sample_rate):
        self.last_warning = None
        if sample_rate != 16000:
            from scipy.signal import resample_poly
            divisor = gcd(sample_rate, 16000)
            samples = resample_poly(samples, 16000 // divisor, sample_rate // divisor)
        samples = np.asarray(samples, dtype=np.float32)
        # Quiet chunks are excluded before ASR; Silero VAD is also enabled below.
        if not len(samples) or float(np.sqrt(np.mean(samples ** 2))) < 0.002:
            return []
        if getattr(self, 'zipformer', False):
            # Each pause-delimited browser chunk gets its own decoder state.
            stream = self.model.create_stream()
            stream.accept_waveform(16000, samples)
            self.model.decode_stream(stream)
            text = ' '.join(stream.result.text.split())
            return [{'start': 0., 'end': len(samples)/16000, 'text': text}] if text else []
        segments, _ = self.model.transcribe(
            samples, language="vi", beam_size=5, best_of=1, temperature=0,
            condition_on_previous_text=False, vad_filter=True,
            vad_parameters={"threshold": 0.6, "min_speech_duration_ms": 300,
                            "min_silence_duration_ms": 500, "speech_pad_ms": 300},
            no_speech_threshold=0.6, log_prob_threshold=-1.0, compression_ratio_threshold=2.2,
            repetition_penalty=1.15, no_repeat_ngram_size=3, max_new_tokens=128,
            # PhoWhisper was fine-tuned for text decoding; forcing timestamp
            # tokens can lose words and yield near-zero segment durations.
            # Use the audio/VAD chunk time instead of asking it for word times.
            without_timestamps=True, word_timestamps=False)
        accepted = []
        for segment in segments:
            if unreliable_segment(segment):
                self.last_warning = "Đã bỏ kết quả chép lời độ tin cậy thấp hoặc lặp bất thường"
                continue
            accepted.append({"start": segment.start, "end": segment.end, "text": segment.text.strip()})
        return accepted
