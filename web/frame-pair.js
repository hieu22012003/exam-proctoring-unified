/* One immutable result/image pair. Never draw old landmarks on a newer live video. */
(function(root) {
  class FramePair {
    constructor() { this.current = null; this.generation = 0; }
    reset() {
      this.generation++;
      this.current?.bitmap.close?.();
      this.current = null;
    }
    async accept(blob, tracks, decode, valid = () => true) {
      const generation = this.generation;
      const bitmap = await decode(blob);
      if (generation !== this.generation || !valid()) { bitmap.close?.(); return false; }
      const previous = this.current;
      this.current = {bitmap, tracks};
      previous?.bitmap.close?.();
      return true;
    }
    paint(canvas) {
      if (!this.current) return null;
      const {bitmap,tracks} = this.current;
      if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
        canvas.width = bitmap.width; canvas.height = bitmap.height;
      }
      canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      return tracks;
    }
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {FramePair};
  else root.ExamWatchFrames = {FramePair};
})(typeof window !== 'undefined' ? window : {});
