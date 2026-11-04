/**
 * MapLibre WebGL2 Custom Layer: 3D Ocean Cutaway (PRD_FRONTEND §3.1b, §7).
 * Renders the top face at sunk depth, side walls from /section, contours,
 * 20°C isotherm line, soft shadow, perimeter outline, and exploded layers mode.
 */

import type { CustomLayerInterface, CustomRenderMethod, Map as MapLibreMap } from "maplibre-gl";
import type { MapResponse, SectionResponse } from "../../lib/api/client";
import {
  depthToZ,
  getDepthTicks,
  getVisibleWalls,
  getWallColorRange,
  screenPointToGridCell,
  type WallId,
} from "./depthScale";
import {
  createGridTexture,
  createLutTexture,
  createProgram,
} from "./webglUtils";

export type MatrixType = Parameters<CustomRenderMethod>[1];

export interface CutawayLayerOptions {
  region: {
    lat_min: number;
    lat_max: number;
    lon_min: number;
    lon_max: number;
  };
  availableDepths: number[];
  colorRanges?: { depth_m: number; vmin: number; vmax: number }[];
  isDark?: boolean;
  dotMode?: boolean;
}

export class CutawayLayer implements CustomLayerInterface {
  id = "ocean-cutaway-3d";
  type: "custom" = "custom";
  renderingMode: "3d" = "3d";

  private map: MapLibreMap | null = null;
  private gl: WebGL2RenderingContext | null = null;

  // Shader programs
  private topFaceProgram: WebGLProgram | null = null;
  private wallProgram: WebGLProgram | null = null;
  private lineProgram: WebGLProgram | null = null;

  // Buffers
  private topFaceVAO: WebGLVertexArrayObject | null = null;
  private topFaceVBO: WebGLBuffer | null = null;

  private wallVAO: WebGLVertexArrayObject | null = null;
  private wallVBO: WebGLBuffer | null = null;

  private lineVAO: WebGLVertexArrayObject | null = null;
  private lineVBO: WebGLBuffer | null = null;

  // Textures
  private lutTexture: WebGLTexture | null = null;
  private topFaceTexture: WebGLTexture | null = null;
  private wallTextures: Record<WallId, WebGLTexture | null> = {
    south: null,
    north: null,
    west: null,
    east: null,
  };

  // State
  private region: CutawayLayerOptions["region"];
  private availableDepths: number[];
  private maxDepth_m: number;
  private colorRanges?: { depth_m: number; vmin: number; vmax: number }[];
  private isDark: boolean;

  private currentDepth_m = 100.0;
  private currentZ = 0.0;
  private targetZ = 0.0;
  private heightMercator = 0.05; // 1/3 of region width
  private animating = false;

  private smoothed = false;
  private exploded = false;
  private showContours = true;
  private dotMode = true;
  private activeField: "pred" | "glorys" | "error" = "pred";
  private lutBalanceTexture: WebGLTexture | null = null;
  private topFaceGridSize: [number, number] = [240, 120];
  private wallGridSizes: Record<WallId, [number, number]> = {
    south: [240, 15],
    north: [240, 15],
    west: [120, 15],
    east: [120, 15],
  };
  private vminTop = 15.0;
  private vmaxTop = 30.0;
  private vminWall = 2.0;
  private vmaxWall = 32.0;

  // Section data caches
  private sectionResponses: Partial<Record<WallId, SectionResponse>> = {};
  private latestMatrix: number[] | null = null;

  constructor(options: CutawayLayerOptions) {
    this.region = options.region;
    this.availableDepths = options.availableDepths;
    this.maxDepth_m = Math.max(...options.availableDepths, 1000);
    this.colorRanges = options.colorRanges;
    this.isDark = options.isDark ?? true;
    this.dotMode = options.dotMode ?? true;

    // Block height is 1/3 of region width in Mercator coordinates
    const widthMerc = (this.region.lon_max - this.region.lon_min) / 360.0;
    this.heightMercator = Math.max(0.01, widthMerc / 3.0);

    const wallRange = getWallColorRange(options.colorRanges);
    this.vminWall = wallRange.vmin;
    this.vmaxWall = wallRange.vmax;
  }

  onAdd(map: MapLibreMap, gl: WebGL2RenderingContext | WebGLRenderingContext): void {
    this.map = map;
    // Verify WebGL2
    if (!(gl instanceof WebGL2RenderingContext)) {
      console.warn("CutawayLayer requires WebGL2; falling back.");
      return;
    }
    this.gl = gl;

    this.initShaders();
    this.initBuffers();
    this.lutTexture = createLutTexture(gl, "thermal");
    this.lutBalanceTexture = createLutTexture(gl, "balance");
  }

  setDepth(depth_m: number, immediate: boolean = false): void {
    this.currentDepth_m = depth_m;
    this.targetZ = depthToZ(depth_m, this.maxDepth_m, this.heightMercator);

    if (immediate) {
      this.currentZ = this.targetZ;
      this.animating = false;
    } else {
      this.animating = true;
    }
    this.map?.triggerRepaint();
  }

  setMapData(mapData: MapResponse, vmin: number, vmax: number): void {
    if (!this.gl) return;
    this.vminTop = vmin;
    this.vmaxTop = vmax;
    this.activeField = mapData.field === "error" ? "error" : mapData.field === "glorys" ? "glorys" : "pred";

    if (this.topFaceTexture) {
      this.gl.deleteTexture(this.topFaceTexture);
      this.topFaceTexture = null;
    }

    const width = mapData.lon.length;
    const height = mapData.lat.length;
    if (width > 0 && height > 0) {
      this.topFaceGridSize = [width, height];
    }
    this.topFaceTexture = createGridTexture(
      this.gl,
      width,
      height,
      mapData.values,
      mapData.coverage,
      vmin,
      vmax,
      this.smoothed
    );
    this.map?.triggerRepaint();
  }

  setSectionData(wallId: WallId, section: SectionResponse): void {
    this.sectionResponses[wallId] = section;
    if (!this.gl) return;

    if (this.wallTextures[wallId]) {
      this.gl.deleteTexture(this.wallTextures[wallId]);
      this.wallTextures[wallId] = null;
    }

    // Section shape: [depths][positions]
    const width = section.positions.length;
    const height = section.depths_m.length;
    if (width > 0 && height > 0) {
      this.wallGridSizes[wallId] = [width, height];
    }

    const wallMin = section.field === "error" ? this.vminTop : this.vminWall;
    const wallMax = section.field === "error" ? this.vmaxTop : this.vmaxWall;

    this.wallTextures[wallId] = createGridTexture(
      this.gl,
      width,
      height,
      section.values,
      section.coverage,
      wallMin,
      wallMax,
      this.smoothed
    );
    this.map?.triggerRepaint();
  }

  setSmoothed(smoothed: boolean): void {
    this.smoothed = smoothed;
    this.map?.triggerRepaint();
  }

  setExploded(exploded: boolean): void {
    this.exploded = exploded;
    this.map?.triggerRepaint();
  }

  setShowContours(show: boolean): void {
    this.showContours = show;
    this.map?.triggerRepaint();
  }

  setIsDark(isDark: boolean): void {
    this.isDark = isDark;
    this.map?.triggerRepaint();
  }

  setDotMode(dotMode: boolean): void {
    this.dotMode = dotMode;
    this.map?.triggerRepaint();
  }

  getSnappedCellAtScreenPoint(
    screenX: number,
    screenY: number,
    lats: number[],
    lons: number[],
    viewportWidth: number,
    viewportHeight: number
  ): { lat: number; lon: number } | null {
    if (!this.latestMatrix) return null;
    return screenPointToGridCell(
      screenX,
      screenY,
      this.latestMatrix,
      this.currentZ,
      lats,
      lons,
      viewportWidth,
      viewportHeight
    );
  }

  render: CustomRenderMethod = (gl, matrix) => {
    if (!this.gl || !(gl instanceof WebGL2RenderingContext) || !this.map) return;
    this.latestMatrix = Array.from(matrix as unknown as ArrayLike<number>);

    // Update dive easing
    if (this.animating) {
      const diff = this.targetZ - this.currentZ;
      if (Math.abs(diff) < 0.0001) {
        this.currentZ = this.targetZ;
        this.animating = false;
      } else {
        this.currentZ += diff * 0.15; // smooth ~300ms ease
        this.map.triggerRepaint();
      }
    }

    this.gl.enable(this.gl.DEPTH_TEST);
    this.gl.depthFunc(this.gl.LEQUAL);
    this.gl.enable(this.gl.BLEND);
    this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);

    // Compute Mercator bounding box coordinates
    const mercMinX = (this.region.lon_min + 180) / 360.0;
    const mercMaxX = (this.region.lon_max + 180) / 360.0;
    const mercMinY = latToMercatorY(this.region.lat_max); // Note: MapLibre Y is inverted (0 = North)
    const mercMaxY = latToMercatorY(this.region.lat_min);

    // 1. Render surface perimeter outline at z = 0
    this.renderPerimeterLines(matrix, mercMinX, mercMaxX, mercMinY, mercMaxY, 0.0, [0.3, 0.6, 0.8, 0.6]);

    // 2. Render sunk depth perimeter outline
    if (Math.abs(this.currentZ) > 0.001) {
      this.renderPerimeterLines(
        matrix,
        mercMinX,
        mercMaxX,
        mercMinY,
        mercMaxY,
        this.currentZ,
        [0.4, 0.8, 1.0, 0.8]
      );
    }

    // 3. Render Top Face (or exploded sheets)
    if (this.exploded) {
      this.renderExplodedSheets(matrix, mercMinX, mercMaxX, mercMinY, mercMaxY);
    } else {
      this.renderTopFace(matrix, mercMinX, mercMaxX, mercMinY, mercMaxY, this.currentZ, 1.0);
    }

    // 4. Render the two visible side walls
    const bearing = this.map.getBearing();
    const visibleWalls = getVisibleWalls(bearing);

    for (const wallId of visibleWalls) {
      this.renderWall(matrix, wallId, mercMinX, mercMaxX, mercMinY, mercMaxY);
    }

    // 5. Render soft bottom shadow
    const floorZ = -this.heightMercator;
    this.renderShadow(matrix, mercMinX, mercMaxX, mercMinY, mercMaxY, floorZ);
  }

  private renderTopFace(
    matrix: MatrixType,
    x0: number,
    x1: number,
    y0: number,
    y1: number,
    z: number,
    opacity: number
  ): void {
    if (!this.gl || !this.topFaceProgram || !this.topFaceVAO || !this.topFaceTexture || !this.lutTexture)
      return;

    this.gl.useProgram(this.topFaceProgram);

    // Update vertex quad at depth z
    // 2 triangles: (x0, y0), (x1, y0), (x0, y1), (x1, y1)
    const verts = new Float32Array([
      // pos (x, y, z), uv (u, v), normal (nx, ny, nz)
      x0, y0, z, 0.0, 1.0, 0.0, 0.0, 1.0,
      x1, y0, z, 1.0, 1.0, 0.0, 0.0, 1.0,
      x0, y1, z, 0.0, 0.0, 0.0, 0.0, 1.0,

      x0, y1, z, 0.0, 0.0, 0.0, 0.0, 1.0,
      x1, y0, z, 1.0, 1.0, 0.0, 0.0, 1.0,
      x1, y1, z, 1.0, 0.0, 0.0, 0.0, 1.0,
    ]);

    this.gl.bindVertexArray(this.topFaceVAO);
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.topFaceVBO);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, verts, this.gl.DYNAMIC_DRAW);

    const activeLut = this.activeField === "error" ? (this.lutBalanceTexture ?? this.lutTexture) : this.lutTexture;

    // Set uniforms
    const uMatrix = this.gl.getUniformLocation(this.topFaceProgram, "u_matrix");
    const uDataTex = this.gl.getUniformLocation(this.topFaceProgram, "u_data_tex");
    const uLutTex = this.gl.getUniformLocation(this.topFaceProgram, "u_lut_tex");
    const uVmin = this.gl.getUniformLocation(this.topFaceProgram, "u_vmin");
    const uVmax = this.gl.getUniformLocation(this.topFaceProgram, "u_vmax");
    const uOpacity = this.gl.getUniformLocation(this.topFaceProgram, "u_opacity");
    const uContours = this.gl.getUniformLocation(this.topFaceProgram, "u_show_contours");
    const uIsDark = this.gl.getUniformLocation(this.topFaceProgram, "u_is_dark");
    const uDotMode = this.gl.getUniformLocation(this.topFaceProgram, "u_dot_mode");
    const uGridSize = this.gl.getUniformLocation(this.topFaceProgram, "u_grid_size");

    this.gl.uniformMatrix4fv(uMatrix, false, matrix as unknown as Float32List);
    this.gl.uniform1f(uVmin, this.vminTop);
    this.gl.uniform1f(uVmax, this.vmaxTop);
    this.gl.uniform1f(uOpacity, opacity);
    this.gl.uniform1i(uContours, this.showContours && this.activeField !== "error" ? 1 : 0);
    this.gl.uniform1i(uIsDark, this.isDark ? 1 : 0);
    this.gl.uniform1i(uDotMode, this.dotMode ? 1 : 0);
    this.gl.uniform2f(uGridSize, this.topFaceGridSize[0], this.topFaceGridSize[1]);

    this.gl.activeTexture(this.gl.TEXTURE0);
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.topFaceTexture);
    this.gl.uniform1i(uDataTex, 0);

    this.gl.activeTexture(this.gl.TEXTURE1);
    this.gl.bindTexture(this.gl.TEXTURE_2D, activeLut);
    this.gl.uniform1i(uLutTex, 1);

    this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
    this.gl.bindVertexArray(null);
  }

  private renderExplodedSheets(
    matrix: MatrixType,
    x0: number,
    x1: number,
    y0: number,
    y1: number
  ): void {
    // 4 sample levels across the column
    const ticks = getDepthTicks(this.availableDepths);
    const depthsToShow = ticks.slice(0, 4).map((t) => t.depth_m);
    if (!depthsToShow.includes(this.currentDepth_m)) {
      depthsToShow.push(this.currentDepth_m);
    }

    for (const d of depthsToShow) {
      const z = depthToZ(d, this.maxDepth_m, this.heightMercator);
      const isSelected = Math.abs(d - this.currentDepth_m) < 0.1;
      const opacity = isSelected ? 1.0 : 0.6;
      this.renderTopFace(matrix, x0, x1, y0, y1, z, opacity);
      this.renderPerimeterLines(matrix, x0, x1, y0, y1, z, [0.3, 0.7, 0.9, opacity]);
    }
  }

  private renderWall(
    matrix: MatrixType,
    wallId: WallId,
    x0: number,
    x1: number,
    y0: number,
    y1: number
  ): void {
    const wallTex = this.wallTextures[wallId];
    if (!this.gl || !this.wallProgram || !this.wallVAO || !wallTex || !this.lutTexture) return;

    this.gl.useProgram(this.wallProgram);

    // Wall coordinates: from current sunk depth currentZ down to floor (-heightMercator)
    const zTop = this.currentZ;
    const zBottom = -this.heightMercator;
    const topRatio = Math.sqrt(Math.max(0, this.currentDepth_m)) / Math.sqrt(this.maxDepth_m);

    let wx0 = x0, wy0 = y0, wx1 = x1, wy1 = y1;
    let normal = [0, -1, 0]; // default south normal

    if (wallId === "south") {
      // Along bottom latitude: from west to east at lat_min (y1)
      wx0 = x0; wy0 = y1;
      wx1 = x1; wy1 = y1;
      normal = [0, 1, 0]; // facing south (+Y in Mercator screen space)
    } else if (wallId === "north") {
      // Along top latitude: from west to east at lat_max (y0)
      wx0 = x1; wy0 = y0;
      wx1 = x0; wy1 = y0;
      normal = [0, -1, 0];
    } else if (wallId === "west") {
      // Along west longitude: from south (y1) to north (y0)
      wx0 = x0; wy0 = y1;
      wx1 = x0; wy1 = y0;
      normal = [-1, 0, 0];
    } else if (wallId === "east") {
      // Along east longitude: from south (y1) to north (y0)
      wx0 = x1; wy0 = y1;
      wx1 = x1; wy1 = y0;
      normal = [1, 0, 0];
    }

    // Quad vertices: 2 triangles with top UV sliced to topRatio
    const verts = new Float32Array([
      // pos (x, y, z), uv (u, v), normal
      wx0, wy0, zTop,    0.0, topRatio, normal[0], normal[1], normal[2],
      wx1, wy1, zTop,    1.0, topRatio, normal[0], normal[1], normal[2],
      wx0, wy0, zBottom, 0.0, 1.0,      normal[0], normal[1], normal[2],

      wx0, wy0, zBottom, 0.0, 1.0,      normal[0], normal[1], normal[2],
      wx1, wy1, zTop,    1.0, topRatio, normal[0], normal[1], normal[2],
      wx1, wy1, zBottom, 1.0, 1.0,      normal[0], normal[1], normal[2],
    ]);

    this.gl.bindVertexArray(this.wallVAO);
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.wallVBO);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, verts, this.gl.DYNAMIC_DRAW);

    const activeLut = this.activeField === "error" ? (this.lutBalanceTexture ?? this.lutTexture) : this.lutTexture;
    const wallMin = this.activeField === "error" ? this.vminTop : this.vminWall;
    const wallMax = this.activeField === "error" ? this.vmaxTop : this.vmaxWall;
    const wallGrid = this.wallGridSizes[wallId];

    const uMatrix = this.gl.getUniformLocation(this.wallProgram, "u_matrix");
    const uSecTex = this.gl.getUniformLocation(this.wallProgram, "u_section_tex");
    const uLutTex = this.gl.getUniformLocation(this.wallProgram, "u_lut_tex");
    const uVmin = this.gl.getUniformLocation(this.wallProgram, "u_vmin");
    const uVmax = this.gl.getUniformLocation(this.wallProgram, "u_vmax");
    const uOpacity = this.gl.getUniformLocation(this.wallProgram, "u_opacity");
    const uContours = this.gl.getUniformLocation(this.wallProgram, "u_show_contours");
    const uIsDark = this.gl.getUniformLocation(this.wallProgram, "u_is_dark");
    const uDotMode = this.gl.getUniformLocation(this.wallProgram, "u_dot_mode");
    const uWallGrid = this.gl.getUniformLocation(this.wallProgram, "u_wall_grid");

    this.gl.uniformMatrix4fv(uMatrix, false, matrix as unknown as Float32List);
    this.gl.uniform1f(uVmin, wallMin);
    this.gl.uniform1f(uVmax, wallMax);
    this.gl.uniform1f(uOpacity, 1.0);
    this.gl.uniform1i(uContours, this.showContours && this.activeField !== "error" ? 1 : 0);
    this.gl.uniform1i(uIsDark, this.isDark ? 1 : 0);
    this.gl.uniform1i(uDotMode, this.dotMode ? 1 : 0);
    this.gl.uniform2f(uWallGrid, wallGrid[0], wallGrid[1]);

    this.gl.activeTexture(this.gl.TEXTURE0);
    this.gl.bindTexture(this.gl.TEXTURE_2D, wallTex);
    this.gl.uniform1i(uSecTex, 0);

    this.gl.activeTexture(this.gl.TEXTURE1);
    this.gl.bindTexture(this.gl.TEXTURE_2D, activeLut);
    this.gl.uniform1i(uLutTex, 1);

    this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
    this.gl.bindVertexArray(null);
  }

  private renderPerimeterLines(
    matrix: MatrixType,
    x0: number,
    x1: number,
    y0: number,
    y1: number,
    z: number,
    color: number[]
  ): void {
    if (!this.gl || !this.lineProgram || !this.lineVAO) return;

    this.gl.useProgram(this.lineProgram);

    const lines = new Float32Array([
      x0, y0, z,  x1, y0, z,
      x1, y0, z,  x1, y1, z,
      x1, y1, z,  x0, y1, z,
      x0, y1, z,  x0, y0, z,
    ]);

    this.gl.bindVertexArray(this.lineVAO);
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.lineVBO);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, lines, this.gl.DYNAMIC_DRAW);

    const uMatrix = this.gl.getUniformLocation(this.lineProgram, "u_matrix");
    const uColor = this.gl.getUniformLocation(this.lineProgram, "u_color");

    this.gl.uniformMatrix4fv(uMatrix, false, matrix as unknown as Float32List);
    this.gl.uniform4fv(uColor, color);

    this.gl.drawArrays(this.gl.LINES, 0, 8);
    this.gl.bindVertexArray(null);
  }

  private renderShadow(
    matrix: MatrixType,
    x0: number,
    x1: number,
    y0: number,
    y1: number,
    z: number
  ): void {
    if (!this.gl || !this.lineProgram || !this.topFaceVAO) return;

    // A slightly expanded floor quad with dark transparent tint
    const pad = (x1 - x0) * 0.05;
    const sx0 = x0 - pad;
    const sx1 = x1 + pad;
    const sy0 = y0 - pad;
    const sy1 = y1 + pad;

    this.gl.useProgram(this.lineProgram);

    const verts = new Float32Array([
      sx0, sy0, z, 0, 0, 0, 0, 1,
      sx1, sy0, z, 0, 0, 0, 0, 1,
      sx0, sy1, z, 0, 0, 0, 0, 1,
      sx0, sy1, z, 0, 0, 0, 0, 1,
      sx1, sy0, z, 0, 0, 0, 0, 1,
      sx1, sy1, z, 0, 0, 0, 0, 1,
    ]);

    this.gl.bindVertexArray(this.topFaceVAO);
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.topFaceVBO);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, verts, this.gl.DYNAMIC_DRAW);

    const uMatrix = this.gl.getUniformLocation(this.lineProgram, "u_matrix");
    const uColor = this.gl.getUniformLocation(this.lineProgram, "u_color");

    this.gl.uniformMatrix4fv(uMatrix, false, matrix as unknown as Float32List);
    this.gl.uniform4fv(uColor, [0.0, 0.0, 0.0, 0.35]);

    this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
    this.gl.bindVertexArray(null);
  }

  private initShaders(): void {
    if (!this.gl) return;

    const vsCommon = `#version 300 es
      layout(location = 0) in vec3 a_pos;
      layout(location = 1) in vec2 a_uv;
      layout(location = 2) in vec3 a_normal;

      uniform mat4 u_matrix;

      out vec2 v_uv;
      out vec3 v_normal;

      void main() {
        v_uv = a_uv;
        v_normal = a_normal;
        gl_Position = u_matrix * vec4(a_pos, 1.0);
      }
    `;

    const fsTopFace = `#version 300 es
      precision highp float;

      in vec2 v_uv;
      in vec3 v_normal;

      uniform sampler2D u_data_tex;
      uniform sampler2D u_lut_tex;
      uniform float u_vmin;
      uniform float u_vmax;
      uniform float u_opacity;
      uniform int u_show_contours;
      uniform int u_is_dark;
      uniform int u_dot_mode;
      uniform vec2 u_grid_size;

      out vec4 fragColor;

      void main() {
        vec4 data = texture(u_data_tex, v_uv);
        int cov = int(data.g * 255.0 + 0.5);

        vec3 bg = (u_is_dark == 1) ? vec3(0.028, 0.051, 0.071) : vec3(0.945, 0.961, 0.976);
        vec2 cell = fract(v_uv * u_grid_size) - 0.5;
        float dist = length(cell);
        float aa = clamp(fwidth(dist), 0.03, 0.14);

        if (cov == 0) {
          if (u_dot_mode == 1) {
            float landMask = smoothstep(0.24 + aa, 0.24 - aa, dist) * 0.82;
            vec3 landDot = (u_is_dark == 1) ? vec3(0.494, 0.404, 0.298) : vec3(0.580, 0.541, 0.486);
            fragColor = vec4(mix(bg, landDot, landMask), u_opacity);
            return;
          }
          if (u_is_dark == 1) {
            fragColor = vec4(0.12, 0.15, 0.20, u_opacity);
          } else {
            fragColor = vec4(0.90, 0.88, 0.85, u_opacity);
          }
          return;
        }

        if (cov == 2) {
          if (u_dot_mode == 1) {
            float ring = smoothstep(0.37 + aa, 0.37 - aa, dist) * smoothstep(0.19 - aa, 0.19 + aa, dist);
            vec3 ringCol = (u_is_dark == 1) ? vec3(0.58, 0.64, 0.72) : vec3(0.28, 0.33, 0.41);
            fragColor = vec4(mix(bg, ringCol, ring * 0.92), u_opacity);
            return;
          }
          vec2 coord = gl_FragCoord.xy;
          float stripe = mod(coord.x + coord.y, 8.0);
          if (stripe < 2.5) {
            fragColor = vec4(0.55, 0.60, 0.65, 0.75 * u_opacity);
          } else {
            fragColor = vec4(0.20, 0.25, 0.30, 0.40 * u_opacity);
          }
          return;
        }

        float norm = (data.r * 256.0 + data.b) / 257.0;
        float temp = u_vmin + norm * (u_vmax - u_vmin);
        vec4 col = texture(u_lut_tex, vec2(norm, 0.5));
        float isoStrength = 0.0;

        if (u_show_contours == 1) {
          float c = abs(fract(temp * 0.5) - 0.5);
          float fw = fwidth(temp * 0.5);
          if (fw > 0.0001) {
            float contour = smoothstep(fw * 1.4, 0.0, c);
            col.rgb = mix(col.rgb, vec3(0.08, 0.08, 0.08), contour * 0.35);
          }

          float isoDiff = abs(temp - 20.0);
          float fwIso = fwidth(temp);
          if (fwIso > 0.0001) {
            isoStrength = smoothstep(fwIso * 1.8, 0.0, isoDiff);
            col.rgb = mix(col.rgb, vec3(1.0, 0.95, 0.35), isoStrength * 0.85);
          }
        }

        if (u_dot_mode == 1) {
          float dotMask = smoothstep(0.38 + aa, 0.38 - aa, dist);
          float core = smoothstep(0.18, 0.0, dist) * 0.11;
          vec3 litDot = min(vec3(1.0), col.rgb + core);
          float maskWithIso = max(dotMask, isoStrength * 0.85);
          fragColor = vec4(mix(bg, litDot, maskWithIso), col.a * u_opacity);
          return;
        }

        fragColor = vec4(col.rgb, col.a * u_opacity);
      }
    `;

    const fsWall = `#version 300 es
      precision highp float;

      in vec2 v_uv;
      in vec3 v_normal;

      uniform sampler2D u_section_tex;
      uniform sampler2D u_lut_tex;
      uniform float u_vmin;
      uniform float u_vmax;
      uniform float u_opacity;
      uniform int u_show_contours;
      uniform int u_is_dark;
      uniform int u_dot_mode;
      uniform vec2 u_wall_grid;

      out vec4 fragColor;

      void main() {
        vec4 data = texture(u_section_tex, v_uv);
        int cov = int(data.g * 255.0 + 0.5);
        vec3 bg = (u_is_dark == 1) ? vec3(0.025, 0.045, 0.065) : vec3(0.92, 0.94, 0.96);

        if (cov == 0) {
          if (u_dot_mode == 1) {
            fragColor = vec4(bg, u_opacity);
          } else {
            fragColor = vec4(0.1, 0.14, 0.18, u_opacity);
          }
          return;
        }

        float norm = (data.r * 256.0 + data.b) / 257.0;
        float temp = u_vmin + norm * (u_vmax - u_vmin);
        vec4 col = texture(u_lut_tex, vec2(norm, 0.5));

        // Shading: darken slightly with depth
        float depthFactor = 1.0 - 0.25 * v_uv.y;
        col.rgb *= depthFactor;
        float isoStrength = 0.0;

        if (u_show_contours == 1) {
          float c = abs(fract(temp * 0.5) - 0.5);
          float fw = fwidth(temp * 0.5);
          if (fw > 0.0001) {
            float contour = smoothstep(fw * 1.4, 0.0, c);
            col.rgb = mix(col.rgb, vec3(0.05, 0.05, 0.05), contour * 0.4);
          }

          float isoDiff = abs(temp - 20.0);
          float fwIso = fwidth(temp);
          if (fwIso > 0.0001) {
            isoStrength = smoothstep(fwIso * 1.8, 0.0, isoDiff);
            col.rgb = mix(col.rgb, vec3(1.0, 0.95, 0.35), isoStrength * 0.85);
          }
        }

        if (u_dot_mode == 1) {
          float colDist = abs(fract(v_uv.x * u_wall_grid.x) - 0.5);
          float aaX = clamp(fwidth(colDist), 0.03, 0.16);
          float colMask = smoothstep(0.36 + aaX, 0.36 - aaX, colDist);
          float mask = max(colMask, isoStrength * 0.85);
          fragColor = vec4(mix(bg, col.rgb, mask), col.a * u_opacity);
          return;
        }

        fragColor = vec4(col.rgb, col.a * u_opacity);
      }
    `;

    const vsLine = `#version 300 es
      layout(location = 0) in vec3 a_pos;
      uniform mat4 u_matrix;
      void main() {
        gl_Position = u_matrix * vec4(a_pos, 1.0);
      }
    `;

    const fsLine = `#version 300 es
      precision mediump float;
      uniform vec4 u_color;
      out vec4 fragColor;
      void main() {
        fragColor = u_color;
      }
    `;

    this.topFaceProgram = createProgram(this.gl, vsCommon, fsTopFace);
    this.wallProgram = createProgram(this.gl, vsCommon, fsWall);
    this.lineProgram = createProgram(this.gl, vsLine, fsLine);
  }

  private initBuffers(): void {
    if (!this.gl) return;

    // Top Face VAO & VBO
    this.topFaceVAO = this.gl.createVertexArray();
    this.topFaceVBO = this.gl.createBuffer();
    this.gl.bindVertexArray(this.topFaceVAO);
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.topFaceVBO);
    // 8 floats per vertex: pos(3), uv(2), normal(3)
    const stride = 8 * 4;
    this.gl.enableVertexAttribArray(0);
    this.gl.vertexAttribPointer(0, 3, this.gl.FLOAT, false, stride, 0);
    this.gl.enableVertexAttribArray(1);
    this.gl.vertexAttribPointer(1, 2, this.gl.FLOAT, false, stride, 3 * 4);
    this.gl.enableVertexAttribArray(2);
    this.gl.vertexAttribPointer(2, 3, this.gl.FLOAT, false, stride, 5 * 4);
    this.gl.bindVertexArray(null);

    // Wall VAO & VBO
    this.wallVAO = this.gl.createVertexArray();
    this.wallVBO = this.gl.createBuffer();
    this.gl.bindVertexArray(this.wallVAO);
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.wallVBO);
    this.gl.enableVertexAttribArray(0);
    this.gl.vertexAttribPointer(0, 3, this.gl.FLOAT, false, stride, 0);
    this.gl.enableVertexAttribArray(1);
    this.gl.vertexAttribPointer(1, 2, this.gl.FLOAT, false, stride, 3 * 4);
    this.gl.enableVertexAttribArray(2);
    this.gl.vertexAttribPointer(2, 3, this.gl.FLOAT, false, stride, 5 * 4);
    this.gl.bindVertexArray(null);

    // Line VAO & VBO
    this.lineVAO = this.gl.createVertexArray();
    this.lineVBO = this.gl.createBuffer();
    this.gl.bindVertexArray(this.lineVAO);
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.lineVBO);
    this.gl.enableVertexAttribArray(0);
    this.gl.vertexAttribPointer(0, 3, this.gl.FLOAT, false, 3 * 4, 0);
    this.gl.bindVertexArray(null);
  }

  onRemove(): void {
    if (!this.gl) return;

    if (this.topFaceProgram) this.gl.deleteProgram(this.topFaceProgram);
    if (this.wallProgram) this.gl.deleteProgram(this.wallProgram);
    if (this.lineProgram) this.gl.deleteProgram(this.lineProgram);

    if (this.topFaceVAO) this.gl.deleteVertexArray(this.topFaceVAO);
    if (this.topFaceVBO) this.gl.deleteBuffer(this.topFaceVBO);
    if (this.wallVAO) this.gl.deleteVertexArray(this.wallVAO);
    if (this.wallVBO) this.gl.deleteBuffer(this.wallVBO);
    if (this.lineVAO) this.gl.deleteVertexArray(this.lineVAO);
    if (this.lineVBO) this.gl.deleteBuffer(this.lineVBO);

    if (this.lutTexture) this.gl.deleteTexture(this.lutTexture);
    if (this.lutBalanceTexture) this.gl.deleteTexture(this.lutBalanceTexture);
    if (this.topFaceTexture) this.gl.deleteTexture(this.topFaceTexture);

    for (const id of Object.keys(this.wallTextures) as WallId[]) {
      const tex = this.wallTextures[id];
      if (tex) this.gl.deleteTexture(tex);
    }
  }
}

/**
 * Standard Web Mercator Y projection [0, 1]
 */
function latToMercatorY(lat: number): number {
  const sin = Math.sin((lat * Math.PI) / 180.0);
  const y = 0.5 - (0.25 * Math.log((1 + sin) / (1 - sin))) / Math.PI;
  return Math.max(0.0, Math.min(1.0, y));
}
