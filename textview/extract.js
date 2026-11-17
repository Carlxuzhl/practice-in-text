// Visible-element extractor: the text view of the rendered page. Evaluated in the page via Playwright; returns a plain
// JSON object.
//
// Allowed inputs (whitelist): Text node data, input value and shown placeholder, layout boxes,
// computed style, paint order from elementsFromPoint, SVG shape geometry.
// Never read: aria-*, data-*, class, id, role, title, alt, name, href, src, tag names as output,
// router state, app state, scroll metadata, or any list of legal actions. A field's own scroll
// offset is used only to work out which of its lines are painted; it never reaches the output.
// The one exception to src: the address of a mask or a small picture keys the table of names given
// by eye in custom_icons.json, and is dropped there, as a shape is.
//
// The only page mutation is a temporary adopted stylesheet so that elementsFromPoint also reports
// pointer-events:none layers; it is removed before returning and fires no DOM mutation.
(opts) => {
  opts = opts || {};
  const VW = window.innerWidth;
  const VH = window.innerHeight;
  const NON_TEXT_PARENTS = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'TITLE', 'OPTION', 'TEXTAREA']);
  const OPAQUE = 0.8;

  const styles = new Map();
  const cs = (el) => {
    let s = styles.get(el);
    if (!s) { s = getComputedStyle(el); styles.set(el, s); }
    return s;
  };

  // Any CSS colour syntax (rgb, oklch, lab, color-mix...) resolved by the browser itself.
  const swatch = document.createElement('canvas').getContext('2d', { willReadFrequently: true });
  const colorMemo = new Map();
  const parseColor = (c) => {
    if (!c) return { r: 0, g: 0, b: 0, a: 0 };
    let v = colorMemo.get(c);
    if (!v) {
      swatch.clearRect(0, 0, 1, 1);
      swatch.fillStyle = '#000';
      swatch.fillStyle = c;
      swatch.fillRect(0, 0, 1, 1);
      const d = swatch.getImageData(0, 0, 1, 1).data;
      v = { r: d[0], g: d[1], b: d[2], a: d[3] / 255 };
      colorMemo.set(c, v);
    }
    return v;
  };

  const opacityMemo = new Map();
  const chainOpacity = (el) => {
    if (!el) return 1;
    if (opacityMemo.has(el)) return opacityMemo.get(el);
    const v = parseFloat(cs(el).opacity || '1') * chainOpacity(el.parentElement);
    opacityMemo.set(el, v);
    return v;
  };

  const luminance = ({ r, g, b }) => {
    const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
  };

  const colorName = (c) => {
    const r = c.r / 255, g = c.g / 255, b = c.b / 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b), l = (max + min) / 2, d = max - min;
    if (d < 0.18 || l < 0.16 || l > 0.96) return l < 0.3 ? 'black' : l < 0.72 ? 'gray' : l < 0.93 ? 'lightgray' : 'white';
    let h = max === r ? ((g - b) / d + 6) % 6 : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
    h *= 60;
    if (h < 15 || h >= 345) return 'red';
    if (h < 45) return 'orange';
    if (h < 70) return 'yellow';
    if (h < 165) return 'green';
    if (h < 200) return 'cyan';
    if (h < 255) return 'blue';
    if (h < 300) return 'purple';
    return 'pink';
  };

  // Solid colour actually painted behind el at (x, y), or null when an image makes it unknown.
  // Paint order decides this: an overlay's DOM ancestors say nothing about what is behind it, and
  // translucent layers tint whatever opaque layer they sit on.
  const backdropAt = (x, y, el) => {
    const stack = document.elementsFromPoint(x, y);
    let i = stack.indexOf(el);
    if (i < 0) i = Math.max(0, stack.findIndex((e) => e.contains(el)));
    const layers = [];
    for (; i < stack.length; i++) {
      const e = stack[i];
      const s = cs(e);
      if (['IMG', 'VIDEO', 'CANVAS'].includes(e.tagName) || (s.backgroundImage && s.backgroundImage !== 'none')) return null;
      const c = parseColor(s.backgroundColor);
      const a = c.a * chainOpacity(e);
      if (a < 0.01) continue;
      layers.push({ ...c, a });
      if (a >= 0.95) {
        const out = layers.pop();
        for (const t of layers.reverse()) for (const k of ['r', 'g', 'b']) out[k] = t[k] * t.a + out[k] * (1 - t.a);
        return out;
      }
    }
    return null;
  };

  // A translucent colour as it reads on screen, given what it is painted over.
  const over = (c, bd) => (c.a >= 0.99 || !bd ? c : {
    r: c.r * c.a + bd.r * (1 - c.a), g: c.g * c.a + bd.g * (1 - c.a), b: c.b * c.a + bd.b * (1 - c.a), a: 1,
  });

  // Screen pixels per layout pixel. The simulator scales its whole screen with a transform, so
  // clientWidth, computed sizes and scroll offsets are not in the units of getBoundingClientRect.
  const scaleOf = (el) => {
    const q = el.getBoundingClientRect();
    return { sx: el.offsetWidth ? q.width / el.offsetWidth : 1, sy: el.offsetHeight ? q.height / el.offsetHeight : 1 };
  };

  const formsContainingBlock = (s) =>
    s.transform !== 'none' || s.perspective !== 'none' || s.filter !== 'none' ||
    /paint|layout|strict|content/.test(s.contain || '') || /transform/.test(s.willChange || '');

  // Intersection of the viewport and every ancestor box that clips el, plus the first
  // ancestor that truncates with an ellipsis or line clamp.
  const clipOf = (el) => {
    let l = 0, t = 0, r = VW, b = VH, ellipsis = null;
    let pos = cs(el).position;
    const own = cs(el);
    if (own.textOverflow === 'ellipsis' || own.webkitLineClamp !== 'none') ellipsis = el;
    if (own.overflowX !== 'visible' || own.overflowY !== 'visible') {
      const q = el.getBoundingClientRect();
      if (own.overflowX !== 'visible') { l = Math.max(l, q.left); r = Math.min(r, q.right); }
      if (own.overflowY !== 'visible') { t = Math.max(t, q.top); b = Math.min(b, q.bottom); }
    }
    for (let a = el.parentElement; a; a = a.parentElement) {
      const s = cs(a);
      const cb = formsContainingBlock(s);
      const inChain = pos === 'fixed' ? cb : pos === 'absolute' ? (s.position !== 'static' || cb) : true;
      if (!inChain) continue;
      if (s.overflowX !== 'visible' || s.overflowY !== 'visible') {
        const q = a.getBoundingClientRect();
        const { sx, sy } = scaleOf(a);
        if (s.overflowX !== 'visible') { l = Math.max(l, q.left + a.clientLeft * sx); r = Math.min(r, q.left + (a.clientLeft + a.clientWidth) * sx); }
        if (s.overflowY !== 'visible') { t = Math.max(t, q.top + a.clientTop * sy); b = Math.min(b, q.top + (a.clientTop + a.clientHeight) * sy); }
        if (!ellipsis && (s.textOverflow === 'ellipsis' || s.webkitLineClamp !== 'none')) ellipsis = a;
      }
      pos = s.position;
    }
    return { l, t, r, b, ellipsis };
  };

  // Quarter turns applied to el or its near ancestors. A chevron turned by CSS points where it is
  // drawn, not where its geometry says.
  const quarterTurns = (el) => {
    let deg = 0;
    for (let a = el, i = 0; a && i < 3; a = a.parentElement, i++) {
      const s = cs(a);
      if (s.rotate && s.rotate !== 'none') deg += parseFloat(s.rotate);
      const m = /^matrix\(([^,]+), ([^,]+)/.exec(s.transform || '');
      if (m) deg += Math.atan2(parseFloat(m[2]), parseFloat(m[1])) * 180 / Math.PI;
    }
    return ((Math.round(deg / 90) * 90) % 360 + 360) % 360;
  };

  const hiddenByClip = (el) => {
    for (let a = el; a; a = a.parentElement) {
      const s = cs(a);
      if (s.clip && /rect\(0(px)?[ ,]+0(px)?[ ,]+0(px)?[ ,]+0(px)?\)/.test(s.clip)) return true;
      if (s.clipPath && /inset\(100%/.test(s.clipPath)) return true;
    }
    return false;
  };

  const paintsOver = (e) => {
    const s = cs(e);
    const op = chainOpacity(e);
    if (op < 0.05) return false;
    if (['IMG', 'VIDEO', 'CANVAS', 'IFRAME'].includes(e.tagName)) return op >= OPAQUE;
    if (s.backdropFilter && s.backdropFilter !== 'none' && /blur/.test(s.backdropFilter)) return true;
    if (s.backgroundImage && s.backgroundImage !== 'none') return op >= OPAQUE;
    return parseColor(s.backgroundColor).a * op >= OPAQUE;
  };

  // True when nothing opaque is painted above target at (x, y).
  const unoccluded = (x, y, target) => {
    if (x < 0 || y < 0 || x >= VW || y >= VH) return false;
    const stack = document.elementsFromPoint(x, y);
    let stop = stack.indexOf(target);
    if (stop < 0) {
      stop = stack.findIndex((e) => e.contains(target) || target.contains(e));
      if (stop < 0) return false;
    }
    for (let i = 0; i < stop; i++) if (paintsOver(stack[i])) return false;
    return true;
  };

  // Fraction of a glyph box inside the clip on each axis. Glyph boxes can be taller than a tight
  // line box, so visibility is judged by overlap rather than containment.
  const overlapX = (q, clip) => (Math.min(q.right, clip.r) - Math.max(q.left, clip.l)) / Math.max(q.right - q.left, 0.01);
  const overlapY = (q, clip) => (Math.min(q.bottom, clip.b) - Math.max(q.top, clip.t)) / Math.max(q.bottom - q.top, 0.01);
  const inlineWhole = (rects, clip) => rects.every((q) => overlapX(q, clip) >= 0.98 && overlapY(q, clip) >= 0.6);

  const measure = document.createElement('canvas').getContext('2d');

  const elements = [];
  const auditNodes = [];
  const push = (item, node) => {
    elements.push(item);
    if (opts.audit) auditNodes.push(node);
  };

  const textStyle = (parent, box) => {
    const s = cs(parent);
    const bd = backdropAt((box.l + box.r) / 2, (box.t + box.b) / 2, parent);
    const out = { color: colorName(over(parseColor(s.webkitTextFillColor || s.color), bd)) };
    if (parseInt(s.fontWeight, 10) >= 600) out.bold = true;
    if (bd) {
      const name = colorName(bd);
      if (!['white', 'lightgray', 'black'].includes(name)) out.bg = name;
    }
    if (chainOpacity(parent) < 0.65) out.dim = true;
    return out;
  };

  const transformText = (t, s) => {
    if (s.textTransform === 'uppercase') return t.toUpperCase();
    if (s.textTransform === 'lowercase') return t.toLowerCase();
    return t;
  };

  const emitText = (node) => {
    const parent = node.parentElement;
    if (!parent || NON_TEXT_PARENTS.has(parent.tagName)) return;
    const raw = node.data;
    if (!raw || !raw.trim()) return;
    const s = cs(parent);
    if (s.visibility !== 'visible' || parseFloat(s.fontSize) < 1) return;
    if (chainOpacity(parent) < 0.05 || hiddenByClip(parent)) return;
    const fill = parseColor(s.webkitTextFillColor || s.color);
    if (fill.a < 0.05) return;
    const range = document.createRange();
    range.selectNodeContents(node);
    const rects = [...range.getClientRects()].filter((q) => q.width > 0.5 && q.height > 0.5);
    if (!rects.length) return;
    const clip = clipOf(parent);
    const samples = [];
    for (const q of rects) {
      const y = (q.top + q.bottom) / 2;
      for (const f of [0.1, 0.5, 0.9]) {
        const x = q.left + q.width * f;
        if (x >= clip.l && x <= clip.r && y >= clip.t && y <= clip.b) samples.push([x, y]);
      }
    }
    if (samples.length) {
      const bd = backdropAt(samples[0][0], samples[0][1], parent);
      if (bd) {
        const a = luminance(fill), b = luminance(bd);
        if ((Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05) < 1.08) return;
      }
    }
    const seen = samples.map(([x, y]) => unoccluded(x, y, parent));
    const allClear = samples.length > 0 && seen.every(Boolean);
    let text = null;
    let box = null;
    if (allClear && inlineWhole(rects, clip)) {
      text = raw;
      box = rects.reduce((u, q) => ({
        l: Math.min(u.l, q.left), t: Math.min(u.t, q.top), r: Math.max(u.r, q.right), b: Math.max(u.b, q.bottom),
      }), { l: 1e9, t: 1e9, r: -1e9, b: -1e9 });
    } else {
      // Keep exactly the characters that are painted and uncovered.
      const kept = [];
      let cap = 0;
      for (let i = 0; i < raw.length && cap < 4000; ) {
        const cp = raw.codePointAt(i);
        const len = cp > 0xffff ? 2 : 1;
        range.setStart(node, i);
        range.setEnd(node, i + len);
        const q = range.getClientRects()[0];
        i += len;
        cap += 1;
        if (!q || q.width < 0.1) { if (kept.length && /\s/.test(raw[i - len])) kept.push({ ch: ' ', q: null }); continue; }
        if (overlapX(q, clip) < 0.6 || overlapY(q, clip) < 0.6) continue;
        if (!unoccluded((q.left + q.right) / 2, (q.top + q.bottom) / 2, parent)) continue;
        kept.push({ ch: String.fromCodePoint(cp), q });
      }
      let truncated = false;
      if (clip.ellipsis && kept.filter((k) => k.q && k.ch.trim()).length < [...raw.replace(/\s+/g, '')].length) {
        // The ellipsis takes the end of the last line; glyphs that no longer fit are not painted.
        measure.font = s.font || `${s.fontSize} ${s.fontFamily}`;
        const limit = clip.r - measure.measureText('\u2026').width * scaleOf(clip.ellipsis).sx + 0.5;
        const lastTop = Math.max(...kept.filter((k) => k.q).map((k) => k.q.top));
        for (let i = kept.length - 1; i >= 0; i--) {
          const k = kept[i];
          if (!k.q) { kept.splice(i, 1); continue; }
          if (k.q.top >= lastTop - 1 && k.q.right > limit) kept.splice(i, 1);
          else break;
        }
        truncated = true;
      }
      const ink = kept.filter((k) => k.q);
      if (!ink.length) return;
      text = kept.map((k) => k.ch).join('') + (truncated ? '\u2026' : '');
      box = ink.reduce((u, k) => ({
        l: Math.min(u.l, k.q.left), t: Math.min(u.t, k.q.top), r: Math.max(u.r, k.q.right), b: Math.max(u.b, k.q.bottom),
      }), { l: 1e9, t: 1e9, r: -1e9, b: -1e9 });
    }
    text = transformText(text, s).replace(/\s+/g, ' ').trim();
    if (!text) return;
    // Icon fonts paint a glyph, not the characters in the DOM.
    if (/^[\uE000-\uF8FF\s]+$/.test(text) || /icon|symbols|material/i.test(s.fontFamily)) {
      push({ role: 'icon', box, ...textStyle(parent, box) }, parent);
      return;
    }
    push({ role: 'text', text, box, ...textStyle(parent, box) }, node);
  };

  const boxVisible = (el, box) => {
    const s = cs(el);
    if (s.visibility !== 'visible' || chainOpacity(el) < 0.05 || hiddenByClip(el)) return null;
    const clip = clipOf(el);
    const l = Math.max(box.left, clip.l), t = Math.max(box.top, clip.t);
    const r = Math.min(box.right, clip.r), b = Math.min(box.bottom, clip.b);
    if (r - l < 3 || b - t < 3) return null;
    const pts = [[0.5, 0.5], [0.25, 0.5], [0.75, 0.5], [0.5, 0.25], [0.5, 0.75]];
    const clear = pts.some(([fx, fy]) => unoccluded(l + (r - l) * fx, t + (b - t) * fy, el));
    return clear ? { l, t, r, b } : null;
  };

  const svgShape = (svg) => {
    const parts = [];
    for (const n of svg.querySelectorAll('path,circle,rect,line,polyline,polygon,ellipse')) {
      const tag = n.tagName.toLowerCase();
      const keys = { path: ['d'], circle: ['cx', 'cy', 'r'], rect: ['x', 'y', 'width', 'height', 'rx', 'ry'],
        line: ['x1', 'y1', 'x2', 'y2'], polyline: ['points'], polygon: ['points'], ellipse: ['cx', 'cy', 'rx', 'ry'] }[tag];
      parts.push(tag + ':' + keys.map((k) => (n.getAttribute(k) || '')).join(','));
    }
    return parts.join('|');
  };

  // Icons usually colour their shapes, and the svg root's own fill is just the black default.
  const dominantFill = (svg, bd) => {
    const paint = (v) => {
      if (!v || v === 'none' || v.startsWith('url')) return null;
      const c = parseColor(v);
      return c.a > 0.1 ? c : null;
    };
    const shapes = svg.querySelectorAll('path,circle,rect,line,polyline,polygon,ellipse');
    for (const n of shapes) {
      const s = cs(n);
      const c = paint(s.fill) || (parseFloat(s.strokeWidth) > 0 ? paint(s.stroke) : null);
      if (c) return colorName(over(c, bd));
    }
    return shapes.length ? null : colorName(over(parseColor(cs(svg).color), bd));
  };

  const emitSvg = (svg) => {
    const box = svg.getBoundingClientRect();
    if (box.width < 3 || box.height < 3) return;
    const vis = boxVisible(svg, box);
    if (!vis) return;
    const hasText = !!svg.querySelector('text');
    const shape = svgShape(svg);
    if (!shape && hasText) return;
    const color = dominantFill(svg, backdropAt((vis.l + vis.r) / 2, (vis.t + vis.b) / 2, svg));
    if (!color) return;  // laid out but painted in nothing, like the tick of an unselected row
    const big = box.width > 96 || box.height > 96;
    const item = { role: big ? 'image' : 'icon', box: vis, color };
    if (opts.iconTable && opts.iconTable[shape]) item.name = opts.iconTable[shape];
    else if (opts.collectShapes) item.shape = shape;
    const turn = quarterTurns(svg);
    if (turn) item.turn = turn;
    if (chainOpacity(svg) < 0.65) item.dim = true;
    push(item, svg);
  };

  const borderPainted = (s, side) => parseFloat(s[`border${side}Width`]) > 0 && s[`border${side}Style`] !== 'none' &&
    parseColor(s[`border${side}Color`]).a > 0.1;

  // A field has no text nodes to measure, so its layout is worked out here. Only what lies inside
  // the visible box counts, or a value scrolled out of view would reach a policy.
  const paintedFieldText = (el, s, vis, shown) => {
    if (el.tagName === 'SELECT') return shown;
    measure.font = s.font || `${s.fontSize} ${s.fontFamily}`;
    const width = el.clientWidth - parseFloat(s.paddingLeft) - parseFloat(s.paddingRight);
    if (el.tagName === 'INPUT') {
      if (measure.measureText(shown).width <= width) return shown;
      let head = '';
      let out = '';
      for (const ch of shown) {
        const x0 = measure.measureText(head).width - el.scrollLeft;
        head += ch;
        if (x0 >= -1 && measure.measureText(head).width - el.scrollLeft <= width + 1) out += ch;
      }
      return out;
    }
    const lines = [];
    for (const para of shown.split('\n')) {
      let cur = '';
      for (const ch of para) {
        if (cur && measure.measureText(cur + ch).width > width) { lines.push(cur); cur = ''; }
        cur += ch;
      }
      lines.push(cur);
    }
    const { sy } = scaleOf(el);
    const lh = (parseFloat(s.lineHeight) || parseFloat(s.fontSize) * 1.2) * sy;
    const y0 = el.getBoundingClientRect().top + (el.clientTop + parseFloat(s.paddingTop) - el.scrollTop) * sy;
    return lines.filter((_, i) => Math.min(y0 + (i + 1) * lh, vis.b) - Math.max(y0 + i * lh, vis.t) >= 0.6 * lh).join(' ');
  };

  // A native checkbox or radio paints its own state, so :checked here is what the screen shows. A
  // restyled one is painted from CSS like any other small box.
  const emitNativeMark = (el) => {
    const box = el.getBoundingClientRect();
    if (box.width < 6 || box.height < 6) return;
    const s = cs(el);
    if (s.appearance === 'none') { emitMark(el, s, box); return; }
    const vis = boxVisible(el, box);
    if (!vis) return;
    const filled = el.matches(':checked');
    const color = !filled ? 'gray' : s.accentColor === 'auto' ? 'blue' : colorName(parseColor(s.accentColor));
    push({ role: 'mark', box: vis, round: el.type === 'radio', color, filled }, el);
  };

  const emitField = (el) => {
    const box = el.getBoundingClientRect();
    const vis = boxVisible(el, box);
    if (!vis) return;
    const s = cs(el);
    const value = el.tagName === 'SELECT' ? (el.selectedOptions[0] ? el.selectedOptions[0].textContent : '') : el.value;
    const masked = el.type === 'password';
    const shown = value ? (masked ? '\u2022'.repeat(value.length) : value) : (el.placeholder || '');
    const ink = value ? parseColor(s.webkitTextFillColor || s.color) : parseColor(getComputedStyle(el, '::placeholder').color);
    const text = (ink.a < 0.05 || !shown ? '' : paintedFieldText(el, s, vis, shown)).replace(/\s+/g, ' ').trim();
    // With no text and no box of its own there is nothing on screen to report.
    if (!text && parseColor(s.backgroundColor).a < 0.1 && !borderPainted(s, 'Top') && !borderPainted(s, 'Bottom')) return;
    const item = { role: 'field', text, box: vis };
    if (!value) item.empty = true;
    if (chainOpacity(el) < 0.65) item.dim = true;
    push(item, el);
  };

  const rounded = (s, el) => {
    const v = s.borderTopLeftRadius;
    if (v.endsWith('%')) return parseFloat(v) >= 40;
    return parseFloat(v) >= (el.offsetHeight || el.getBoundingClientRect().height) / 2 - 1.5;
  };

  // A pill with one round knob. On/off comes from where the knob is painted.
  const asSwitch = (el, s, box) => {
    const w = box.width, h = box.height;
    if (h < 12 || h > 48 || w / h < 1.3 || w / h > 3.2) return false;
    if (!rounded(s, el) || parseColor(s.backgroundColor).a < 0.1) return false;
    const kids = el.children;
    if (kids.length !== 1) return false;
    const k = kids[0];
    const kb = k.getBoundingClientRect();
    const ks = cs(k);
    if (kb.width < h * 0.5 || kb.height > h + 1 || Math.abs(kb.width - kb.height) > 3) return false;
    if (!rounded(ks, k)) return false;
    const vis = boxVisible(el, box);
    if (!vis) return true;
    const on = (kb.left + kb.right) / 2 > (box.left + box.right) / 2;
    push({ role: 'switch', on, box: vis, color: colorName(parseColor(s.backgroundColor)) }, el);
    return true;
  };

  // A small empty box or ring: an unticked checkbox, radio, swatch or dot.
  const emitMark = (el, s, box) => {
    const w = box.width, h = box.height;
    if (w < 8 || h < 8 || w > 40 || h > 40 || Math.abs(w - h) > 3) return false;
    if (el.childElementCount > 1 || (el.textContent || '').trim()) return false;
    if (el.querySelector('svg,img')) return false;
    const border = borderPainted(s, 'Top');
    const bg = parseColor(s.backgroundColor);
    // A ring whose inside matches what is behind it reads as empty, whatever paints that inside.
    const behind = el.parentElement ? backdropAt((box.left + box.right) / 2, (box.top + box.bottom) / 2, el.parentElement) : null;
    const filled = bg.a > 0.1 && !(behind && Math.abs(luminance(bg) - luminance(behind)) < 0.02 && colorName(bg) === colorName(behind));
    if (!border && !filled) return false;
    const vis = boxVisible(el, box);
    if (!vis) return false;
    const item = { role: 'mark', box: vis, round: rounded(s, el), color: colorName(filled ? bg : parseColor(s.borderTopColor)) };
    item.filled = filled;
    push(item, el);
    return true;
  };

  // A small picture may be an icon. Like a mask's, its address only keys the icon name table in textview.py
  // and never reaches the output on its own, so a picture outside the table stays a bare icon. The
  // whole path is the key, because file names repeat across apps.
  const emitImage = (el, box) => {
    if (box.width < 8 || box.height < 8) return;
    if (el.tagName === 'IMG' && (!el.complete || el.naturalWidth === 0)) return;
    const vis = boxVisible(el, box);
    if (!vis) return;
    const big = box.width > 64 || box.height > 64;
    const item = { role: big ? 'image' : 'icon', box: vis, raster: true };
    const from = (el.currentSrc || cs(el).backgroundImage).replace(/^url\(["']?|["']?\)\s*$/g, '').replace(/^https?:\/\/[^/]+/, '');
    if (opts.collectShapes && !big && from && from !== 'none') item.shape = 'raster:' + from;
    push(item, el);
  };

  // A coloured box cut to an icon's shape by a CSS mask. The mask's file name only keys the icon
  // name table in textview.py and never reaches the output on its own.
  const emitMaskIcon = (el, s, box) => {
    const vis = boxVisible(el, box);
    if (!vis) return;
    const item = { role: 'icon', box: vis, color: colorName(parseColor(s.backgroundColor)) };
    const file = /([^/"']+?)(\?[^/"']*)?["']?\)\s*$/.exec(s.maskImage || s.webkitMaskImage);
    if (opts.collectShapes && file) item.shape = 'mask:' + file[1];
    const turn = quarterTurns(el);
    if (turn) item.turn = turn;
    if (chainOpacity(el) < 0.65) item.dim = true;
    push(item, el);
  };

  // Colour of a glyph built from tiny painted boxes, like a nine-dot menu, or null if el is not one.
  const dotGlyph = (el, box) => {
    if (el.childElementCount < 4 || box.width > 40 || box.height > 40 || (el.textContent || '').trim()) return null;
    let color = null;
    for (const k of el.children) {
      const q = k.getBoundingClientRect();
      color = parseColor(cs(k).backgroundColor);
      if (q.width > 8 || q.height > 8 || color.a < 0.1) return null;
    }
    return color;
  };

  const hasOwnText = (el) => {
    for (const c of el.childNodes) if (c.nodeType === 3 && c.data.trim()) return true;
    return false;
  };

  const walk = (root) => {
    for (let n = root.firstChild; n; n = n.nextSibling) {
      if (n.nodeType === 3) { emitText(n); continue; }
      if (n.nodeType !== 1) continue;
      const tag = n.tagName;
      if (NON_TEXT_PARENTS.has(tag) && tag !== 'TEXTAREA') continue;
      const s = cs(n);
      if (s.display === 'none') continue;
      if (tag === 'INPUT') {
        if (n.type === 'checkbox' || n.type === 'radio') emitNativeMark(n);
        else if (n.type !== 'hidden' && !['range', 'file', 'color'].includes(n.type)) emitField(n);
        continue;
      }
      if (tag === 'TEXTAREA' || tag === 'SELECT') { emitField(n); continue; }
      if (tag === 'svg') {
        emitSvg(n);
        for (const t of n.querySelectorAll('text')) for (const c of t.childNodes) if (c.nodeType === 3) emitText(c);
        continue;
      }
      if (tag === 'IMG' || tag === 'VIDEO' || tag === 'CANVAS') { emitImage(n, n.getBoundingClientRect()); continue; }
      const box = n.getBoundingClientRect();
      const onScreen = box.right > 0 && box.bottom > 0 && box.left < VW && box.top < VH;
      if (onScreen && box.width > 0 && box.height > 0) {
        if (/url\(/.test(s.maskImage || s.webkitMaskImage || '')) { emitMaskIcon(n, s, box); continue; }
        const dots = dotGlyph(n, box);
        if (dots) {
          const vis = boxVisible(n, box);
          if (vis) push({ role: 'icon', name: 'grip', box: vis, color: colorName(dots) }, n);
          continue;
        }
        if (s.backgroundImage && /url\(/.test(s.backgroundImage) && !hasOwnText(n)) emitImage(n, box);
        else if (asSwitch(n, s, box)) continue;
        else if (emitMark(n, s, box)) continue;
      }
      walk(n);
    }
  };

  const sheet = new CSSStyleSheet();
  sheet.replaceSync('*{pointer-events:auto !important}');
  // adoptedStyleSheets is a live array: copy it, or the restore below restores nothing.
  const before = [...document.adoptedStyleSheets];
  document.adoptedStyleSheets = [...before, sheet];
  try {
    walk(document.body);
  } finally {
    document.adoptedStyleSheets = before;
  }
  const restored = !document.adoptedStyleSheets.includes(sheet) && document.adoptedStyleSheets.length === before.length;

  // Diagnostics for environment checks only; render() never shows these to a policy.
  let imgOk = 0, imgBroken = 0;
  for (const img of document.images) {
    const b = img.getBoundingClientRect();
    if (b.width < 8 || b.height < 8 || b.right <= 0 || b.bottom <= 0 || b.left >= VW || b.top >= VH) continue;
    if (cs(img).visibility !== 'visible' || chainOpacity(img) < 0.05) continue;
    if (img.complete && img.naturalWidth > 0) imgOk += 1; else imgBroken += 1;
  }

  // Reading order comes from the screen, never from DOM order.
  const line = (it) => Math.round(it.box.t / 6);
  const order = elements.map((e, i) => i).sort((a, b) =>
    line(elements[a]) - line(elements[b]) || elements[a].box.l - elements[b].box.l);
  let sorted = order.map((i) => elements[i]);
  const sortedNodes = order.map((i) => auditNodes[i]);

  // Join contiguous inline runs (a currency sign and "12.50"), which read as one string on screen.
  const merged = [];
  const mergedNodes = [];
  for (let i = 0; i < sorted.length; i++) {
    const it = sorted[i];
    const prev = merged[merged.length - 1];
    if (prev && prev.role === 'text' && it.role === 'text' && it.color === prev.color && it.bg === prev.bg) {
      const gap = it.box.l - prev.box.r;
      const overlap = Math.min(it.box.b, prev.box.b) - Math.max(it.box.t, prev.box.t);
      const minH = Math.min(it.box.b - it.box.t, prev.box.b - prev.box.t);
      const maxH = Math.max(it.box.b - it.box.t, prev.box.b - prev.box.t);
      if (gap >= -1 && gap < 3 && overlap >= 0.6 * minH && maxH <= 1.7 * minH) {
        prev.text += it.text;
        prev.box = { l: Math.min(prev.box.l, it.box.l), t: Math.min(prev.box.t, it.box.t), r: Math.max(prev.box.r, it.box.r), b: Math.max(prev.box.b, it.box.b) };
        if (opts.audit) mergedNodes[mergedNodes.length - 1].push(sortedNodes[i]);
        continue;
      }
    }
    merged.push(it);
    if (opts.audit) mergedNodes.push([sortedNodes[i]]);
  }
  sorted = merged;
  if (opts.audit) window.__TEXTVIEW_AUDIT_NODES__ = mergedNodes;

  const norm = (b) => [
    Math.round(b.l / VW * 1000), Math.round(b.t / VH * 1000),
    Math.round(b.r / VW * 1000), Math.round(b.b / VH * 1000),
  ];
  return {
    viewport: [VW, VH],
    restored,
    diag: { imgOk, imgBroken },
    elements: sorted.map((it, i) => {
      const { box, ...rest } = it;
      return { id: i + 1, ...rest, bbox: norm(box) };
    }),
  };
}
