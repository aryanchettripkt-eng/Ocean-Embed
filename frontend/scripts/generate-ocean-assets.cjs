// Usage: node scripts/generate-ocean-assets.cjs light.png dark.png fish.png
// Uses Next.js's existing sharp dependency; no new image-processing dependency.
const fs = require('node:fs');
const path = require('node:path');
const sharp = require(require.resolve('sharp', { paths: [path.dirname(require.resolve('next/package.json'))] }));
const out = path.join(__dirname, '../public/landing/ocean');
async function main() {
  const [light, dark, fish] = process.argv.slice(2);
  if (!light || !dark || !fish) throw new Error('Pass light, dark and transparent fish source paths');
  fs.mkdirSync(out, { recursive: true });
  for (const [theme, src] of [['light', light], ['dark', dark]]) {
    for (const width of [640, 1024, 1600]) {
      const file = path.join(out, `${theme}-${width}.webp`);
      await sharp(src).resize(width).webp({ quality: 78 }).toFile(file);
      console.log(path.basename(file), fs.statSync(file).size);
    }
  }
  await sharp(fish).trim().resize(384).webp({ quality: 85 }).toFile(path.join(out, 'fish.webp'));
  await sharp(dark).resize(1200, 630, { fit: 'cover' }).png().toFile(path.join(out, 'og.png'));
}
main().catch(error => { console.error(error); process.exitCode = 1; });
