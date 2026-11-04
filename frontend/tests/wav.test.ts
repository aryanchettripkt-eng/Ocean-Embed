import { describe, expect, it } from "vitest";
import { encodeWav, MAX_WAV_SECONDS, WAV_RATE } from "../src/lib/wav";

describe("encodeWav (voice dive upload format)", () => {
  it("writes a mono 16-bit PCM WAV header at 16 kHz", async () => {
    const blob = encodeWav(new Float32Array([0, 1, -1, 0.5]));
    expect(blob.type).toBe("audio/wav");
    const v = new DataView(await blob.arrayBuffer());
    const ascii = (o: number) => String.fromCharCode(...[0, 1, 2, 3].map((i) => v.getUint8(o + i)));
    expect([ascii(0), ascii(8), ascii(12), ascii(36)]).toEqual(["RIFF", "WAVE", "fmt ", "data"]);
    expect(v.getUint16(20, true)).toBe(1); // PCM
    expect(v.getUint16(22, true)).toBe(1); // mono
    expect(v.getUint32(24, true)).toBe(16_000);
    expect(v.getUint16(34, true)).toBe(16);
    expect(v.getUint32(40, true)).toBe(8);
    expect([0, 1, 2, 3].map((i) => v.getInt16(44 + i * 2, true))).toEqual([0, 32767, -32768, 16383]);
  });

  it("keeps a full 30 s clip inside the backend's 1 MB cap", () => {
    const blob = encodeWav(new Float32Array(MAX_WAV_SECONDS * WAV_RATE));
    expect(blob.size).toBe(960_044);
    expect(blob.size).toBeLessThanOrEqual(1_000_000);
  });
});
