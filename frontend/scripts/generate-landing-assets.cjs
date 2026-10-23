const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const sharp = require(path.join(__dirname, '..', 'node_modules', '.pnpm', 'sharp@0.35.5_@types+node@22.20.5', 'node_modules', 'sharp'));

const outDir = path.join(__dirname, '..', 'public', 'landing');
if (!fs.existsSync(outDir)) {
  fs.mkdirSync(outDir, { recursive: true });
}

// 1. Generate grain.png (128x128 noise PNG <= 6 KB)
function makePngChunk(type, data) {
  function crc32(buf) {
    let table = [];
    for (let i = 0; i < 256; i++) {
      let c = i;
      for (let j = 0; j < 8; j++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
      table[i] = c;
    }
    let crc = 0 ^ (-1);
    for (let i = 0; i < buf.length; i++) crc = (crc >>> 8) ^ table[(crc ^ buf[i]) & 0xff];
    return (crc ^ (-1)) >>> 0;
  }
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const typeBuf = Buffer.from(type, 'ascii');
  const crcBuf = Buffer.alloc(4);
  const toCrc = Buffer.concat([typeBuf, data]);
  crcBuf.writeUInt32BE(crc32(toCrc), 0);
  return Buffer.concat([len, typeBuf, data, crcBuf]);
}

function generateGrain() {
  const w = 128, h = 128;
  const raw = Buffer.alloc(h * (1 + w * 2));
  let ptr = 0;
  for (let y = 0; y < h; y++) {
    raw[ptr++] = 0; // Filter: None
    for (let x = 0; x < w; x++) {
      // Subtle pseudo-random paper texture
      const n = ((x * 47 + y * 73 + (x ^ y) * 23) % 97);
      raw[ptr++] = 30; // Ink gray level
      raw[ptr++] = (n > 65) ? Math.floor((n - 65) * 1.8) : 0; // Alpha
    }
  }
  const compressed = zlib.deflateSync(raw);
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8;
  ihdr[9] = 4; // Grayscale with alpha
  const png = Buffer.concat([
    sig,
    makePngChunk('IHDR', ihdr),
    makePngChunk('IDAT', compressed),
    makePngChunk('IEND', Buffer.alloc(0))
  ]);
  fs.writeFileSync(path.join(outDir, 'grain.png'), png);
  console.log(`grain.png: ${png.length} bytes`);
}

// 2. Generate dither-strip.png (16 frames of 16x16 Bayer matrix threshold mask, 256x16 px)
function generateDitherStrip() {
  const bayer4x4 = [
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5]
  ];
  const frames = 16;
  const frameW = 16;
  const frameH = 16;
  const totalW = frames * frameW;
  const totalH = frameH;

  const raw = Buffer.alloc(totalH * (1 + totalW * 2));
  let ptr = 0;
  for (let y = 0; y < totalH; y++) {
    raw[ptr++] = 0; // Filter: None
    for (let x = 0; x < totalW; x++) {
      const frameIdx = Math.floor(x / frameW);
      const localX = x % frameW;
      const bayerVal = bayer4x4[y % 4][localX % 4];
      const isFilled = bayerVal <= frameIdx;
      raw[ptr++] = 0; // Black mask
      raw[ptr++] = isFilled ? 255 : 0; // Alpha
    }
  }
  const compressed = zlib.deflateSync(raw);
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(totalW, 0);
  ihdr.writeUInt32BE(totalH, 4);
  ihdr[8] = 8;
  ihdr[9] = 4;
  const png = Buffer.concat([
    sig,
    makePngChunk('IHDR', ihdr),
    makePngChunk('IDAT', compressed),
    makePngChunk('IEND', Buffer.alloc(0))
  ]);
  fs.writeFileSync(path.join(outDir, 'dither-strip.png'), png);
  console.log(`dither-strip.png: ${png.length} bytes`);
}

// 3. Generate satellite.svg
function generateSatelliteSvg() {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300" width="400" height="300" fill="none">
    <!-- Orbit dashed ray indicator -->
    <path d="M 40 260 C 120 180 280 120 380 40" stroke="#1E2A2B" stroke-width="1.5" stroke-dasharray="6 6" opacity="0.4"/>
    <!-- Left Solar Array -->
    <g transform="translate(70, 110) rotate(-15)">
      <rect x="0" y="0" width="90" height="42" fill="#E4D8BA" stroke="#1E2A2B" stroke-width="2"/>
      <line x1="30" y1="0" x2="30" y2="42" stroke="#1E2A2B" stroke-width="1.2"/>
      <line x1="60" y1="0" x2="60" y2="42" stroke="#1E2A2B" stroke-width="1.2"/>
      <line x1="0" y1="21" x2="90" y2="21" stroke="#1E2A2B" stroke-width="1.2"/>
      <path d="M 90 21 L 115 21" stroke="#1E2A2B" stroke-width="3"/>
    </g>
    <!-- Right Solar Array -->
    <g transform="translate(235, 68) rotate(-15)">
      <path d="M -25 21 L 0 21" stroke="#1E2A2B" stroke-width="3"/>
      <rect x="0" y="0" width="90" height="42" fill="#E4D8BA" stroke="#1E2A2B" stroke-width="2"/>
      <line x1="30" y1="0" x2="30" y2="42" stroke="#1E2A2B" stroke-width="1.2"/>
      <line x1="60" y1="0" x2="60" y2="42" stroke="#1E2A2B" stroke-width="1.2"/>
      <line x1="0" y1="21" x2="90" y2="21" stroke="#1E2A2B" stroke-width="1.2"/>
    </g>
    <!-- Satellite Body -->
    <g transform="translate(180, 85) rotate(-15)">
      <rect x="0" y="0" width="46" height="56" rx="2" fill="#F6F1E2" stroke="#1E2A2B" stroke-width="2.5"/>
      <rect x="8" y="8" width="30" height="18" fill="#F2B66D" stroke="#1E2A2B" stroke-width="1.5"/>
      <!-- Radiometer / Altimeter sensor dish -->
      <path d="M 23 56 L 23 74" stroke="#1E2A2B" stroke-width="2.5"/>
      <path d="M 6 74 Q 23 88 40 74" stroke="#1E2A2B" stroke-width="2" fill="#E4D8BA"/>
      <!-- Radar beam lines downward -->
      <line x1="16" y1="84" x2="6" y2="108" stroke="#2F6B73" stroke-width="1.5" stroke-dasharray="3 3"/>
      <line x1="23" y1="86" x2="23" y2="114" stroke="#2F6B73" stroke-width="1.5" stroke-dasharray="3 3"/>
      <line x1="30" y1="84" x2="40" y2="108" stroke="#2F6B73" stroke-width="1.5" stroke-dasharray="3 3"/>
    </g>
  </svg>`;
  fs.writeFileSync(path.join(outDir, 'satellite.svg'), svg.trim());
  console.log(`satellite.svg: ${svg.length} bytes`);
}

// 4. Generate argo-float.svg
function generateArgoFloatSvg() {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 420" width="120" height="420" fill="none">
    <!-- Top antenna -->
    <line x1="60" y1="15" x2="60" y2="90" stroke="#1E2A2B" stroke-width="2.5"/>
    <circle cx="60" cy="15" r="4" fill="#1E2A2B"/>
    <!-- CTD sensor head -->
    <rect x="48" y="90" width="24" height="24" rx="2" fill="#E4D8BA" stroke="#1E2A2B" stroke-width="2"/>
    <circle cx="60" cy="102" r="4" fill="#1E2A2B"/>
    <line x1="52" y1="114" x2="68" y2="114" stroke="#1E2A2B" stroke-width="2"/>
    <!-- Collar -->
    <path d="M 42 114 L 78 114 L 74 130 L 46 130 Z" fill="#F2B66D" stroke="#1E2A2B" stroke-width="2"/>
    <!-- Main Pressure Hull (Yellow/Ochre cylinder) -->
    <rect x="44" y="130" width="32" height="180" rx="4" fill="#F2B66D" stroke="#1E2A2B" stroke-width="2.5"/>
    <!-- Technical Hull Bands -->
    <line x1="44" y1="160" x2="76" y2="160" stroke="#1E2A2B" stroke-width="2"/>
    <line x1="44" y1="210" x2="76" y2="210" stroke="#1E2A2B" stroke-width="2"/>
    <line x1="44" y1="260" x2="76" y2="260" stroke="#1E2A2B" stroke-width="2"/>
    <text x="60" y="238" font-family="monospace" font-size="9" fill="#1E2A2B" font-weight="bold" text-anchor="middle" letter-spacing="1">ARGO</text>
    <!-- Hydraulic bladder base -->
    <path d="M 46 310 L 74 310 L 68 340 L 52 340 Z" fill="#E4D8BA" stroke="#1E2A2B" stroke-width="2"/>
    <!-- Damping stabilization disk -->
    <ellipse cx="60" cy="344" rx="26" ry="6" fill="#1E2A2B"/>
    <!-- Downward probe -->
    <line x1="60" y1="344" x2="60" y2="385" stroke="#1E2A2B" stroke-width="2.5"/>
    <circle cx="60" cy="385" r="3" fill="#1F4E57"/>
  </svg>`;
  fs.writeFileSync(path.join(outDir, 'argo-float.svg'), svg.trim());
  console.log(`argo-float.svg: ${svg.length} bytes`);
}

// 5. Generate sky-coast.webp (2400x1100)
async function generateSkyCoast() {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2400 1100" width="2400" height="1100">
    <!-- Paper base -->
    <rect width="2400" height="1100" fill="#EFE6CF"/>
    
    <!-- Sun disc with soft woodblock aura -->
    <circle cx="1680" cy="380" r="140" fill="#E4D8BA" opacity="0.6"/>
    <circle cx="1680" cy="380" r="115" fill="#F2B66D"/>
    
    <!-- Distant mountain ridges (faint paper-shade silhouettes) -->
    <path d="M 900 680 Q 1150 420 1450 680 Q 1650 490 1950 680 L 2400 680 L 2400 850 L 900 850 Z" fill="#E4D8BA" opacity="0.7"/>
    <!-- Middle mountains with ink lines -->
    <path d="M 1200 700 Q 1480 480 1780 720 Q 2000 530 2300 700 L 2400 700 L 2400 850 L 1200 850 Z" fill="#D8CBA7" stroke="#1E2A2B" stroke-width="2"/>
    <path d="M 1480 480 L 1520 700" stroke="#1E2A2B" stroke-width="1.5" stroke-dasharray="4 8" opacity="0.6"/>
    <path d="M 2000 530 L 2040 700" stroke="#1E2A2B" stroke-width="1.5" stroke-dasharray="4 8" opacity="0.6"/>

    <!-- Woodblock cloud trails / kasumi bands -->
    <g fill="#F6F1E2" stroke="#1E2A2B" stroke-width="1.5">
      <path d="M 120 180 C 260 180 340 160 480 160 C 580 160 640 180 760 180 L 760 215 C 620 215 540 195 420 195 C 320 195 240 215 120 215 Z" opacity="0.85"/>
      <path d="M 480 240 C 620 240 700 220 840 220 C 940 220 1020 240 1140 240 L 1140 270 C 1000 270 920 250 820 250 C 720 250 640 270 480 270 Z" opacity="0.85"/>
      <path d="M 1400 260 C 1520 260 1600 240 1740 240 C 1840 240 1920 260 2060 260 L 2060 290 C 1920 290 1840 270 1740 270 C 1640 270 1560 290 1400 290 Z" opacity="0.85"/>
    </g>

    <!-- Coastal shoreline, cliffs and headlands -->
    <path d="M 1400 780 C 1600 760 1750 780 1900 740 C 2050 700 2200 720 2400 690 L 2400 880 L 1400 880 Z" fill="#E4D8BA" stroke="#1E2A2B" stroke-width="2.5"/>
    
    <!-- Coastal Temple / Pagoda on cliff in ink -->
    <g transform="translate(1820, 640)">
      <!-- Tier 1 -->
      <path d="M 10 70 L 70 70 L 60 48 L 20 48 Z" fill="#EFE6CF" stroke="#1E2A2B" stroke-width="2"/>
      <path d="M 0 50 Q 40 38 80 50" stroke="#1E2A2B" stroke-width="3" fill="none"/>
      <!-- Tier 2 -->
      <path d="M 22 48 L 58 48 L 52 30 L 28 30 Z" fill="#EFE6CF" stroke="#1E2A2B" stroke-width="2"/>
      <path d="M 12 32 Q 40 22 68 32" stroke="#1E2A2B" stroke-width="2.5" fill="none"/>
      <!-- Spire -->
      <line x1="40" y1="22" x2="40" y2="4" stroke="#1E2A2B" stroke-width="2"/>
      <circle cx="40" cy="4" r="3" fill="#F2B66D" stroke="#1E2A2B" stroke-width="1.5"/>
    </g>

    <!-- Flying birds in ink -->
    <g stroke="#1E2A2B" stroke-width="2" fill="none">
      <path d="M 680 320 Q 695 310 710 320 Q 725 310 740 320"/>
      <path d="M 730 350 Q 742 342 754 350 Q 766 342 778 350"/>
      <path d="M 790 310 Q 800 302 810 310 Q 820 302 830 310"/>
    </g>

    <!-- Sea boundary line (sea area below left clear for ocean-band / waves) -->
    <line x1="0" y1="880" x2="2400" y2="880" stroke="#1E2A2B" stroke-width="1.5" stroke-dasharray="12 6" opacity="0.3"/>
  </svg>`;

  const buffer = await sharp(Buffer.from(svg))
    .webp({ quality: 75 })
    .toBuffer();
  fs.writeFileSync(path.join(outDir, 'sky-coast.webp'), buffer);
  console.log(`sky-coast.webp: ${buffer.length} bytes (limit 120 KB)`);
}

// 6. Generate tree.webp (900x700, transparent)
async function generateTree() {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 700" width="900" height="700">
    <g fill="none" stroke="#1E2A2B">
      <!-- Main trunk/branch coming from top-right corner -->
      <path d="M 900 20 C 780 40 680 90 560 140 C 440 190 320 270 240 360 C 200 405 160 480 130 540" stroke-width="26" stroke-linecap="round"/>
      <path d="M 900 20 C 780 40 680 90 560 140 C 440 190 320 270 240 360 C 200 405 160 480 130 540" stroke="#E4D8BA" stroke-width="14" stroke-linecap="round"/>
      
      <!-- Sub-branches -->
      <path d="M 620 115 C 540 80 460 70 380 90" stroke-width="12" stroke-linecap="round"/>
      <path d="M 480 165 C 410 170 350 200 290 235" stroke-width="10" stroke-linecap="round"/>
      <path d="M 360 235 C 310 265 270 320 210 350" stroke-width="8" stroke-linecap="round"/>
      <path d="M 260 335 C 210 350 170 410 120 440" stroke-width="7" stroke-linecap="round"/>

      <!-- Stylized Pine Needle Fans (Woodblock style) in ink with soft filling -->
      <g fill="#2F6B73" stroke="#1E2A2B" stroke-width="2" opacity="0.95">
        <!-- Cluster 1 -->
        <path d="M 360 90 C 330 60 280 70 260 110 C 290 125 340 120 360 90 Z"/>
        <path d="M 400 80 C 380 40 330 40 310 80 C 340 95 380 95 400 80 Z"/>
        <!-- Cluster 2 -->
        <path d="M 280 230 C 230 200 190 230 180 270 C 220 285 260 270 280 230 Z"/>
        <path d="M 310 210 C 270 170 220 180 200 230 C 240 245 290 240 310 210 Z"/>
        <!-- Cluster 3 -->
        <path d="M 190 340 C 140 310 100 340 90 380 C 130 395 170 380 190 340 Z"/>
        <!-- Cluster 4 -->
        <path d="M 110 430 C 60 410 30 450 30 490 C 70 500 100 480 110 430 Z"/>
        <path d="M 130 520 C 80 500 50 540 50 580 C 90 590 120 570 130 520 Z"/>
      </g>
    </g>
  </svg>`;

  const buffer = await sharp(Buffer.from(svg))
    .webp({ quality: 80 })
    .toBuffer();
  fs.writeFileSync(path.join(outDir, 'tree.webp'), buffer);
  console.log(`tree.webp: ${buffer.length} bytes (limit 120 KB)`);
}

// 7. Generate seabed.webp (1000x500, transparent)
async function generateSeabed() {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 500" width="1000" height="500">
    <g fill="none">
      <!-- Rocky floor and reef mounds in ink and deep sea tones -->
      <path d="M 0 500 L 0 320 Q 80 300 160 340 Q 260 290 380 360 Q 500 320 620 410 Q 740 380 860 450 L 1000 460 L 1000 500 Z" fill="#14353B" stroke="#1E2A2B" stroke-width="3"/>
      <path d="M 0 500 L 0 380 Q 120 360 220 410 Q 340 370 480 430 L 700 470 L 1000 490 L 1000 500 Z" fill="#1E2A2B" opacity="0.6"/>

      <!-- Coral fans and sea fans (woodblock ink strokes) -->
      <g stroke="#F2B66D" stroke-width="2.5" fill="none">
        <path d="M 140 340 C 120 280 80 250 50 220 M 140 340 C 150 270 130 220 110 190 M 140 340 C 170 280 180 230 180 190 M 140 340 C 190 290 220 260 250 240"/>
        <circle cx="50" cy="220" r="3" fill="#F2B66D"/>
        <circle cx="110" cy="190" r="3" fill="#F2B66D"/>
        <circle cx="180" cy="190" r="3" fill="#F2B66D"/>
        <circle cx="250" cy="240" r="3" fill="#F2B66D"/>
      </g>

      <g stroke="#2F6B73" stroke-width="3" fill="none">
        <!-- Kelp / Sea grass blades curving in current -->
        <path d="M 280 360 C 260 290 310 220 280 160 C 260 120 280 80 270 40"/>
        <path d="M 310 365 C 290 310 340 240 320 180 C 300 140 330 90 310 60"/>
        <path d="M 340 375 C 330 330 370 270 350 220"/>
      </g>

      <!-- School of ocean fish silhouettes swimming eastward in ink -->
      <g fill="#E4D8BA" stroke="#1E2A2B" stroke-width="1.5">
        <!-- Fish 1 -->
        <path d="M 480 220 C 510 210 530 220 540 220 L 550 212 L 546 220 L 550 228 L 540 220 Z"/>
        <!-- Fish 2 -->
        <path d="M 520 185 C 545 177 560 185 570 185 L 578 178 L 575 185 L 578 192 L 570 185 Z"/>
        <!-- Fish 3 -->
        <path d="M 560 230 C 585 222 600 230 610 230 L 618 223 L 615 230 L 618 237 L 610 230 Z"/>
        <!-- Fish 4 -->
        <path d="M 600 195 C 625 188 640 195 650 195 L 658 189 L 655 195 L 658 201 L 650 195 Z"/>
      </g>
    </g>
  </svg>`;

  const buffer = await sharp(Buffer.from(svg))
    .webp({ quality: 80 })
    .toBuffer();
  fs.writeFileSync(path.join(outDir, 'seabed.webp'), buffer);
  console.log(`seabed.webp: ${buffer.length} bytes (limit 120 KB)`);
}

// 8. Generate ocean-band-placeholder.webp (1600x500)
async function generateOceanBand() {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 500" width="1600" height="500">
    <defs>
      <linearGradient id="oceanGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#F2B66D"/>
        <stop offset="12%" stop-color="#E59B56"/>
        <stop offset="28%" stop-color="#CA774E"/>
        <stop offset="45%" stop-color="#6F6D69"/>
        <stop offset="65%" stop-color="#2F6B73"/>
        <stop offset="85%" stop-color="#1F4E57"/>
        <stop offset="100%" stop-color="#14353B"/>
      </linearGradient>
    </defs>
    
    <!-- Thermal gradient field -->
    <rect width="1600" height="500" fill="url(#oceanGrad)"/>

    <!-- Subtle horizontal temperature contour ripples -->
    <g fill="none" stroke="#F6F1E2" stroke-width="1.5" stroke-dasharray="8 6" opacity="0.65">
      <!-- 28°C contour near surface -->
      <path d="M 0 55 Q 400 65 800 50 T 1600 60"/>
      <!-- 24°C contour -->
      <path d="M 0 115 Q 400 130 800 110 T 1600 125"/>
      <!-- 20°C Isotherm (prominent reference contour) -->
      <path d="M 0 180 Q 400 205 800 175 T 1600 190" stroke="#F6F1E2" stroke-width="2.5" stroke-dasharray="10 4" opacity="0.9"/>
      <!-- 16°C contour -->
      <path d="M 0 260 Q 400 275 800 250 T 1600 270"/>
      <!-- 12°C contour -->
      <path d="M 0 350 Q 400 360 800 345 T 1600 355"/>
      <!-- 8°C deep contour -->
      <path d="M 0 430 Q 400 435 800 425 T 1600 430"/>
    </g>

    <!-- Internal wave thermocline displacement curls -->
    <g fill="none" stroke="#E4D8BA" stroke-width="1.2" opacity="0.4">
      <path d="M 200 160 Q 300 220 400 170 Q 500 130 600 180"/>
      <path d="M 900 170 Q 1050 230 1200 180 Q 1350 140 1500 185"/>
    </g>
  </svg>`;

  const buffer = await sharp(Buffer.from(svg))
    .webp({ quality: 80 })
    .toBuffer();
  fs.writeFileSync(path.join(outDir, 'ocean-band-placeholder.webp'), buffer);
  console.log(`ocean-band-placeholder.webp: ${buffer.length} bytes (limit 120 KB)`);
}

// 9. Generate og.png (1200x630)
async function generateOg() {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 630" width="1200" height="630">
    <rect width="1200" height="630" fill="#EFE6CF"/>
    
    <!-- Outer Ink Border -->
    <rect x="24" y="24" width="1152" height="582" fill="none" stroke="#1E2A2B" stroke-width="3"/>
    <rect x="32" y="32" width="1136" height="566" fill="none" stroke="#1E2A2B" stroke-width="1" stroke-dasharray="6 4"/>

    <!-- Sun disc -->
    <circle cx="980" cy="220" r="90" fill="#F2B66D"/>

    <!-- Mountains and waves background silhouettes -->
    <path d="M 550 420 Q 750 260 950 420 Q 1080 320 1200 420 L 1200 600 L 550 600 Z" fill="#E4D8BA" opacity="0.6"/>
    <path d="M 0 470 C 300 450 600 490 900 460 C 1050 445 1150 470 1200 455 L 1200 608 L 0 608 Z" fill="#1F4E57"/>
    <path d="M 0 520 C 350 510 700 530 1200 515 L 1200 608 L 0 608 Z" fill="#14353B"/>

    <!-- Brand Header -->
    <g transform="translate(80, 160)">
      <text x="0" y="0" font-family="monospace" font-size="20" font-weight="bold" fill="#2F6B73" letter-spacing="3">NORTH INDIAN OCEAN · 15 DEPTH LEVELS</text>
      <text x="0" y="80" font-family="monospace" font-size="68" font-weight="900" fill="#1E2A2B" letter-spacing="2">OCEANSCOPE AI</text>
      <text x="0" y="135" font-family="monospace" font-size="30" font-weight="bold" fill="#1F4E57" letter-spacing="6">BENEATH THE SURFACE</text>
      <text x="0" y="195" font-family="monospace" font-size="20" fill="#4A5654">AI-powered 3D subsurface ocean temperature reconstruction</text>
      <text x="0" y="225" font-family="monospace" font-size="20" fill="#4A5654">from satellite surface dynamics and in-situ ARGO observations.</text>
    </g>

    <!-- Badge in bottom left -->
    <g transform="translate(80, 535)">
      <rect x="0" y="0" width="280" height="42" fill="#E4D8BA" stroke="#1E2A2B" stroke-width="2"/>
      <text x="140" y="26" font-family="monospace" font-size="14" font-weight="bold" fill="#1E2A2B" text-anchor="middle" letter-spacing="2">SURFACE TO 1,000 M</text>
    </g>
  </svg>`;

  const buffer = await sharp(Buffer.from(svg))
    .png()
    .toBuffer();
  fs.writeFileSync(path.join(outDir, 'og.png'), buffer);
  console.log(`og.png: ${buffer.length} bytes`);
}

async function run() {
  console.log('Generating landing page assets in', outDir);
  generateGrain();
  generateDitherStrip();
  generateSatelliteSvg();
  generateArgoFloatSvg();
  await generateSkyCoast();
  await generateTree();
  await generateSeabed();
  await generateOceanBand();
  await generateOg();
  console.log('All landing assets generated successfully!');
}

run().catch(err => {
  console.error(err);
  process.exit(1);
});
