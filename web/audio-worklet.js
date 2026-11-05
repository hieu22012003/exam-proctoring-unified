class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(Math.round(sampleRate * 8));
    this.count = 0;
    this.total = 0;
    this.enabled = true;
    this.silentSamples = 0;
    this.hasSpeech = false;
    this.meterSamples = 0;
    this.port.onmessage = ({data}) => {
      if (data === 'flush') {
        this.enabled = false;
        this.emit();
        this.port.postMessage({done: true});
      }
    };
  }
  emit() {
    if (!this.count) return;
    const samples = this.buffer.slice(0, this.count);
    this.port.postMessage({samples, start: this.total / sampleRate, rate: sampleRate}, [samples.buffer]);
    this.total += this.count;
    this.count = 0;
    this.silentSamples = 0;
    this.hasSpeech = false;
  }
  process(inputs, outputs) {
    // Output stays silent: microphone audio never loops back through the speakers.
    const channel = inputs[0]?.[0];
    if (!channel || !this.enabled) return true;
    let energy = 0;
    for (const value of channel) energy += value * value;
    const speaking = Math.sqrt(energy / channel.length) >= 0.004;
    this.meterSamples += channel.length;
    if (this.meterSamples >= sampleRate / 4) {
      this.port.postMessage({level: Math.sqrt(energy / channel.length)});
      this.meterSamples = 0;
    }
    this.hasSpeech = this.hasSpeech || speaking;
    this.silentSamples = speaking ? 0 : this.silentSamples + channel.length;
    for (let i = 0; i < channel.length; i++) {
      this.buffer[this.count++] = channel[i];
      if (this.count === this.buffer.length) this.emit();
    }
    // Prefer sentence pauses to arbitrary cuts. Silero VAD on the server makes
    // the actual speech decision; this RMS gate only selects a useful boundary.
    if (this.hasSpeech && this.count >= sampleRate * 1.5 && this.silentSamples >= sampleRate * 0.6) this.emit();
    return true;
  }
}
registerProcessor('examwatch-capture', CaptureProcessor);
