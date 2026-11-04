/**
 * Turn a browser recording into the only format Muse Voice Transcribe's file endpoint takes:
 * mono 16-bit PCM WAV at 16 kHz (dev.meta.ai/docs/speech-to-text). MediaRecorder gives WebM,
 * MP4 or Ogg, so the clip is decoded and resampled here, in memory. 30 s at 16 kHz is 960 kB,
 * inside the backend's 1 MB cap.
 */

export const WAV_RATE = 16_000;
export const MAX_WAV_SECONDS = 30;

/** Encode mono samples in [-1, 1] as a 16-bit PCM WAV file. */
export function encodeWav(samples: Float32Array, rate = WAV_RATE): Blob {
  const dataLen = samples.length * 2;
  const buf = new ArrayBuffer(44 + dataLen);
  const v = new DataView(buf);
  const ascii = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i));
  };
  ascii(0, "RIFF");
  v.setUint32(4, 36 + dataLen, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  v.setUint32(16, 16, true); // fmt chunk size
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, 1, true); // mono
  v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true); // byte rate
  v.setUint16(32, 2, true); // block align
  v.setUint16(34, 16, true); // bits per sample
  ascii(36, "data");
  v.setUint32(40, dataLen, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

/** Decode a MediaRecorder clip, mix to mono, resample to 16 kHz, cap at 30 s, encode as WAV. */
export async function recordingToWav(clip: Blob): Promise<Blob> {
  const Ctx =
    window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Ctx || typeof OfflineAudioContext === "undefined") throw new Error("Web Audio is not available");
  const ctx = new Ctx();
  let decoded: AudioBuffer;
  try {
    decoded = await ctx.decodeAudioData(await clip.arrayBuffer());
  } finally {
    void ctx.close();
  }
  const seconds = Math.min(decoded.duration, MAX_WAV_SECONDS);
  const frames = Math.max(1, Math.floor(seconds * WAV_RATE));
  // A one-channel offline context mixes the source down to mono while it resamples.
  const offline = new OfflineAudioContext(1, frames, WAV_RATE);
  const src = offline.createBufferSource();
  src.buffer = decoded;
  src.connect(offline.destination);
  src.start();
  const rendered = await offline.startRendering();
  return encodeWav(rendered.getChannelData(0), WAV_RATE);
}
