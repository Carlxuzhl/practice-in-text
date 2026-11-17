// Build the shape -> icon-name table from lucide's published icon geometry. Names describe what the glyph looks like;
// the key is the rendered SVG geometry, never a class or aria attribute.
// Usage: node build_icon_table.mjs <lucide-react/dist/esm/icons> <out.json>
import fs from 'node:fs';
import path from 'node:path';

const dir = process.argv[2];
const out = process.argv[3];
const KEYS = {
  path: ['d'], circle: ['cx', 'cy', 'r'], rect: ['x', 'y', 'width', 'height', 'rx', 'ry'],
  line: ['x1', 'y1', 'x2', 'y2'], polyline: ['points'], polygon: ['points'], ellipse: ['cx', 'cy', 'rx', 'ry'],
};
const table = {};
let n = 0;
for (const f of fs.readdirSync(dir).sort()) {
  if (!f.endsWith('.js')) continue;
  const src = fs.readFileSync(path.join(dir, f), 'utf8');
  const m = /const __iconNode = (\[[\s\S]*?\n\]);/.exec(src);
  if (!m) continue;
  const node = new Function(`return ${m[1]}`)();
  const shape = node.map(([tag, attrs]) => `${tag}:${KEYS[tag].map((k) => attrs[k] ?? '').join(',')}`).join('|');
  const name = f.replace(/\.js$/, '');
  if (!table[shape] || name < table[shape]) table[shape] = name;
  n += 1;
}
fs.writeFileSync(out, JSON.stringify(table));
console.log(`${n} icons, ${Object.keys(table).length} distinct shapes -> ${out}`);
