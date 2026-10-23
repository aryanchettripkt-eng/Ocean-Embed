/**
 * WebGL2 utilities for 3D Cutaway Layer (PRD_FRONTEND §3.1b, §7).
 */

import { THERMAL_LUT, BALANCE_LUT, type RGB } from "../../lib/colormaps";

let cachedWebGL2Support: boolean | null = null;

/**
 * Check if WebGL2 is supported in the current environment.
 */
export function isWebGL2Supported(): boolean {
  if (cachedWebGL2Support !== null) return cachedWebGL2Support;
  if (typeof window === "undefined" || typeof document === "undefined") {
    return false;
  }
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2");
    cachedWebGL2Support = gl !== null && gl !== undefined;
  } catch {
    cachedWebGL2Support = false;
  }
  return cachedWebGL2Support;
}

/**
 * Compile a WebGL shader.
 */
export function createShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader compilation error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

/**
 * Link a WebGL program from vertex and fragment shaders.
 */
export function createProgram(
  gl: WebGL2RenderingContext,
  vertexSource: string,
  fragmentSource: string
): WebGLProgram | null {
  const vs = createShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fs = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  if (!vs || !fs) return null;

  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  // Clean up individual shaders once linked
  gl.deleteShader(vs);
  gl.deleteShader(fs);

  return program;
}

/**
 * Create a 256x1 1D LUT texture from RGB array.
 */
export function createLutTexture(
  gl: WebGL2RenderingContext,
  colormap: "thermal" | "balance" = "thermal"
): WebGLTexture | null {
  const lut: RGB[] = colormap === "balance" ? BALANCE_LUT : THERMAL_LUT;
  const pixels = new Uint8Array(256 * 4);

  for (let i = 0; i < 256; i++) {
    const rgb = lut[i] || [0, 0, 0];
    pixels[i * 4 + 0] = rgb[0];
    pixels[i * 4 + 1] = rgb[1];
    pixels[i * 4 + 2] = rgb[2];
    pixels[i * 4 + 3] = 255;
  }

  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA8,
    256,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    pixels
  );

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  return texture;
}

/**
 * Upload a 2D data grid to a texture.
 * Channels:
 * R: Temperature value normalized to [0, 255] between vmin and vmax
 * G: Coverage flag (0 = land, 1 = valid, 2 = insufficient inputs)
 * B: High-precision fraction byte
 * A: 255
 */
export function createGridTexture(
  gl: WebGL2RenderingContext,
  width: number,
  height: number,
  values: (number | null)[][], // [height][width]
  coverage: number[][] | number[], // [height][width] or 1D [width]
  vmin: number,
  vmax: number,
  smoothed: boolean = false
): WebGLTexture | null {
  const pixels = new Uint8Array(width * height * 4);
  const range = vmax > vmin ? vmax - vmin : 1.0;
  const is2DCov = coverage.length > 0 && Array.isArray(coverage[0]);

  for (let r = 0; r < height; r++) {
    const valRow = values[r] || [];
    const covRow = is2DCov ? (coverage as number[][])[r] || [] : null;
    for (let c = 0; c < width; c++) {
      const idx = (r * width + c) * 4;
      const val = valRow[c];
      const cov = is2DCov ? (covRow ? covRow[c] ?? 0 : 0) : ((coverage as number[])[c] ?? 0);

      if (val === null || val === undefined || isNaN(val) || cov === 0) {
        pixels[idx + 0] = 0;
        pixels[idx + 1] = cov; // coverage 0
        pixels[idx + 2] = 0;
        pixels[idx + 3] = 255;
      } else {
        const norm = Math.max(0.0, Math.min(1.0, (val - vmin) / range));
        const scaled = norm * 65535;
        const high = Math.floor(scaled / 256);
        const low = Math.floor(scaled % 256);

        pixels[idx + 0] = high;
        pixels[idx + 1] = cov; // 1 or 2
        pixels[idx + 2] = low;
        pixels[idx + 3] = 255;
      }
    }
  }

  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA8,
    width,
    height,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    pixels
  );

  const filter = smoothed ? gl.LINEAR : gl.NEAREST;
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  return texture;
}
