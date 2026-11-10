class PcmTap extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Int16Array(512);
    this.offset = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    for (let index = 0; index < channel.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, channel[index]));
      this.buffer[this.offset] = sample < 0 ? sample * 32768 : sample * 32767;
      this.offset += 1;
      if (this.offset === 512) {
        const frame = this.buffer.buffer;
        this.port.postMessage(frame, [frame]);
        this.buffer = new Int16Array(512);
        this.offset = 0;
      }
    }
    return true;
  }
}

registerProcessor('pcm-tap', PcmTap);
