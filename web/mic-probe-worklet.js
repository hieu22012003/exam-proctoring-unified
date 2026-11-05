// A separate, bounded capture for listening to the microphone before ASR.
// It never posts data to the exam-session queue or changes its segmentation.
class MicProbe extends AudioWorkletProcessor {
  constructor() {
    super();
    this.samples = new Float32Array(Math.round(sampleRate * 8));
    this.count = 0;
    this.lastProgress = 0;
    this.done = false;
  }
  process(inputs) {
    if (this.done) return false;
    const channel = inputs[0]?.[0];
    if (!channel) return true;
    const size = Math.min(channel.length, this.samples.length - this.count);
    this.samples.set(channel.subarray(0, size), this.count);
    this.count += size;
    if (this.count === this.samples.length) {
      this.done = true;
      this.port.postMessage({samples:this.samples, rate:sampleRate}, [this.samples.buffer]);
      return false;
    }
    if (this.count - this.lastProgress >= sampleRate / 4) {
      this.port.postMessage({seconds:this.count / sampleRate});
      this.lastProgress = this.count;
    }
    return true;
  }
}
registerProcessor('examwatch-mic-probe', MicProbe);
