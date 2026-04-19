"use client";

/**
 * Streaming PCM16 → Web Audio playback with a master GainNode so we can
 * ramp to silence on barge-in.
 *
 * Chunks arrive at `serverRate` Hz (default 24kHz); we decode into
 * Float32, wrap in an AudioBuffer, and schedule via BufferSourceNode
 * against a playback clock so chunks concatenate without drift.
 */

const FADE_OUT_MS = 5;
const FADE_IN_GUARD_MS = 20;

export class AudioPlayback {
  private ctx: AudioContext | null = null;
  private gain: GainNode | null = null;
  private serverRate: number;
  private nextStartTime: number = 0;
  private sources: Set<AudioBufferSourceNode> = new Set();

  constructor(serverRate: number = 24000) {
    this.serverRate = serverRate;
  }

  async init(): Promise<void> {
    if (this.ctx) return;
    type AudioContextCtor = typeof AudioContext;
    const AC: AudioContextCtor =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: AudioContextCtor }).webkitAudioContext!;
    const ctx = new AC();
    if (ctx.state === "suspended") await ctx.resume();
    const gain = ctx.createGain();
    gain.gain.value = 1;
    gain.connect(ctx.destination);
    this.ctx = ctx;
    this.gain = gain;
    this.nextStartTime = ctx.currentTime;
  }

  /** Queue a PCM16 LE chunk for playback. */
  enqueue(pcm16le: ArrayBuffer): void {
    if (!this.ctx || !this.gain) return;
    const samples = new Int16Array(pcm16le);
    if (samples.length === 0) return;
    const float = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) float[i] = samples[i] / 0x8000;

    // Construct an AudioBuffer at the server's sample rate. The Web Audio
    // mixer will resample to the ctx's native rate on playback.
    const buf = this.ctx.createBuffer(1, float.length, this.serverRate);
    buf.copyToChannel(float, 0);
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.gain);

    const now = this.ctx.currentTime;
    const startAt = Math.max(this.nextStartTime, now);
    src.start(startAt);
    this.nextStartTime = startAt + buf.duration;

    this.sources.add(src);
    src.onended = () => this.sources.delete(src);
  }

  /**
   * Immediately stop all queued audio. Short gain ramp prevents click.
   * Resolves after the fade-in guard so the next enqueue doesn't get
   * chopped off by the restoring ramp.
   */
  async interrupt(): Promise<void> {
    if (!this.ctx || !this.gain) return;
    const now = this.ctx.currentTime;
    const gainParam = this.gain.gain;
    gainParam.cancelScheduledValues(now);
    gainParam.setValueAtTime(gainParam.value, now);
    gainParam.linearRampToValueAtTime(0, now + FADE_OUT_MS / 1000);
    await wait(FADE_OUT_MS + 1);

    for (const src of this.sources) {
      try {
        src.stop();
      } catch {
        // already stopped
      }
    }
    this.sources.clear();
    this.nextStartTime = this.ctx.currentTime;

    // Restore gain for the next turn.
    const restoreNow = this.ctx.currentTime;
    gainParam.setValueAtTime(0, restoreNow);
    gainParam.linearRampToValueAtTime(1, restoreNow + FADE_IN_GUARD_MS / 1000);
  }

  /** True while any scheduled chunks are still playing or pending. */
  isBusy(): boolean {
    if (!this.ctx) return false;
    return this.sources.size > 0 || this.nextStartTime > this.ctx.currentTime;
  }

  async dispose(): Promise<void> {
    if (!this.ctx) return;
    try {
      await this.interrupt();
      await this.ctx.close();
    } catch {
      // ignore
    }
    this.ctx = null;
    this.gain = null;
    this.sources.clear();
  }
}

function wait(ms: number): Promise<void> {
  return new Promise((r) => window.setTimeout(r, ms));
}
