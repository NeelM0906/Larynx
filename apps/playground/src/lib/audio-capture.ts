"use client";

/**
 * getUserMedia → PCM16 @ 16kHz → WS binary frames.
 *
 * Uses ScriptProcessorNode (deprecated but universal) so we don't have
 * to bundle a separate AudioWorklet module. Swap to AudioWorklet once
 * the feature lands and we need the lower latency.
 */

const TARGET_RATE = 16000;
const FRAME_SIZE = 2048;

export interface AudioCaptureHandle {
  stop(): Promise<void>;
  sourceSampleRate: number;
}

export type PCMCallback = (pcm16: ArrayBuffer) => void;

export async function startCapture(onPCM: PCMCallback): Promise<AudioCaptureHandle> {
  if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
    throw new Error(
      "Microphone capture isn't available — needs a secure (HTTPS) origin and a browser with getUserMedia.",
    );
  }

  // Permissive constraints. Safari is strict about unknown keys; keeping
  // to the universally-supported set avoids OverconstrainedError on
  // hardware that doesn't advertise channelCount control.
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  type AudioContextCtor = typeof AudioContext;
  const AC: AudioContextCtor =
    window.AudioContext ??
    (window as unknown as { webkitAudioContext?: AudioContextCtor }).webkitAudioContext!;
  if (!AC) {
    stream.getTracks().forEach((t) => t.stop());
    throw new Error("Web Audio API isn't available in this browser.");
  }
  const ctx = new AC();
  // Safari commonly starts the context in "suspended" even when created
  // from a user gesture. Explicit resume avoids silent capture.
  if (ctx.state === "suspended") {
    try {
      await ctx.resume();
    } catch {
      // non-fatal — if it stays suspended we just won't get audio frames
    }
  }
  const source = ctx.createMediaStreamSource(stream);
  const processor = ctx.createScriptProcessor(FRAME_SIZE, 1, 1);

  const inputRate = ctx.sampleRate;
  const ratio = inputRate / TARGET_RATE;

  processor.onaudioprocess = (event) => {
    const input = event.inputBuffer.getChannelData(0);
    const downsampled = downsample(input, ratio);
    const pcm16 = encodePCM16(downsampled);
    // Force plain ArrayBuffer (TS's ArrayBufferLike union now includes
    // SharedArrayBuffer; WebSocket.send wants the narrow type).
    const out = new ArrayBuffer(pcm16.byteLength);
    new Int16Array(out).set(pcm16);
    onPCM(out);
  };

  source.connect(processor);
  // Must connect to destination for onaudioprocess to fire. Gain-to-zero
  // so we don't get feedback through the loopback.
  const sink = ctx.createGain();
  sink.gain.value = 0;
  processor.connect(sink);
  sink.connect(ctx.destination);

  const stop = async () => {
    try {
      processor.disconnect();
      source.disconnect();
      sink.disconnect();
    } catch {
      // ignore — processor may already be disconnected
    }
    stream.getTracks().forEach((t) => t.stop());
    if (ctx.state !== "closed") {
      try {
        await ctx.close();
      } catch {
        // ignore
      }
    }
  };

  return { stop, sourceSampleRate: inputRate };
}

function downsample(input: Float32Array, ratio: number): Float32Array {
  if (ratio === 1) return input;
  const outLength = Math.floor(input.length / ratio);
  const out = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const idx = i * ratio;
    const lo = Math.floor(idx);
    const hi = Math.min(lo + 1, input.length - 1);
    const t = idx - lo;
    out[i] = input[lo] * (1 - t) + input[hi] * t;
  }
  return out;
}

function encodePCM16(samples: Float32Array): Int16Array {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}
