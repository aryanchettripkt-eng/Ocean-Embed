const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const sharp = require(path.join(__dirname, '..', 'node_modules', '.pnpm', 'sharp@0.35.5_@types+node@22.20.5', 'node_modules', 'sharp'));

const outDir = path.join(__dirname, '..', 'public', 'landing');
const srcDir = path.join(outDir, 'src');

if (!fs.existsSync(outDir)) {
  fs.mkdirSync(outDir, { recursive: true });
}
if (!fs.existsSync(srcDir)) {
  fs.mkdirSync(srcDir, { recursive: true });
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

// 2. Generate dither-strip.png (4x4 Bayer matrix pattern)
function generateDitherStrip() {
  const bayer4x4 = [
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5]
  ];
  const w = 4, h = 4;
  const raw = Buffer.alloc(h * (1 + w * 2));
  let ptr = 0;
  for (let y = 0; y < h; y++) {
    raw[ptr++] = 0;
    for (let x = 0; x < w; x++) {
      raw[ptr++] = 0; // Black
      raw[ptr++] = Math.round((bayer4x4[y][x] / 16) * 255); // Alpha
    }
  }
  const compressed = zlib.deflateSync(raw);
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
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

// 3. Generate og.png (1200x630)
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

// 4. Generate scene-{640,1024,1600,2400}.{avif,webp} from public/landing/src/scene-clean.png
async function generateSceneAssets() {
  const srcFile = path.join(srcDir, 'scene-clean.png');
  if (!fs.existsSync(srcFile)) {
    console.error(`Source image not found: ${srcFile}`);
    process.exit(1);
  }

  const widths = [640, 1024, 1600, 2400];
  for (const w of widths) {
    const h = Math.round(w * 940 / 1673);

    // AVIF q=50
    const avifBuf = await sharp(srcFile)
      .resize(w, h, { kernel: 'lanczos3' })
      .avif({ quality: 50 })
      .toBuffer();
    const avifPath = path.join(outDir, `scene-${w}.avif`);
    fs.writeFileSync(avifPath, avifBuf);
    console.log(`scene-${w}.avif: ${avifBuf.length} bytes (${Math.round(avifBuf.length / 1024)} KB)`);

    // WebP q=72
    const webpBuf = await sharp(srcFile)
      .resize(w, h, { kernel: 'lanczos3' })
      .webp({ quality: 72 })
      .toBuffer();
    const webpPath = path.join(outDir, `scene-${w}.webp`);
    fs.writeFileSync(webpPath, webpBuf);
    console.log(`scene-${w}.webp: ${webpBuf.length} bytes (${Math.round(webpBuf.length / 1024)} KB)`);
  }
}

async function run() {
  console.log('Generating landing page assets in', outDir);
  generateGrain();
  generateDitherStrip();
  await generateOg();
  await generateSceneAssets();
  console.log('All landing assets generated successfully!');
}

run().catch(err => {
  console.error(err);
  process.exit(1);
});
