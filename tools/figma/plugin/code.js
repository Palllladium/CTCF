// CTCF Architecture Diagram Generator — Figma Plugin
// Reads a JSON spec and creates native Figma objects.

figma.showUI(__html__, { width: 520, height: 640 });

figma.ui.onmessage = async (msg) => {
  if (msg.type === 'create') {
    try {
      const spec = JSON.parse(msg.json);
      const count = await createDiagram(spec);
      figma.ui.postMessage({ type: 'done', count });
      figma.notify(`Diagram created: ${count} elements`);
    } catch (e) {
      figma.ui.postMessage({ type: 'error', message: e.message });
      figma.notify('Error: ' + e.message, { error: true });
    }
  }
};

// ─── Color helpers ──────────────────────────────────────────────────

function hexToRgb(hex) {
  if (!hex || hex === 'none') return null;
  hex = hex.replace('#', '');
  // Expand 3-char hex (#fff -> ffffff)
  if (hex.length === 3) {
    hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
  }
  const r = parseInt(hex.substr(0, 2), 16) / 255;
  const g = parseInt(hex.substr(2, 2), 16) / 255;
  const b = parseInt(hex.substr(4, 2), 16) / 255;
  // Guard against NaN from malformed hex
  if (isNaN(r) || isNaN(g) || isNaN(b)) return null;
  return { r, g, b };
}

function solidPaint(hex) {
  const rgb = hexToRgb(hex);
  if (!rgb) return [];
  return [{ type: 'SOLID', color: rgb }];
}

// ─── Font helpers ───────────────────────────────────────────────────

function figmaFontName(style) {
  // style is e.g. "Regular", "Bold", "Semi Bold", "Italic", "Bold Italic"
  return { family: 'Inter', style: style || 'Regular' };
}

function resolveFontStyle(weight, style) {
  // Map SVG-like weight/style to Figma font style string
  const w = String(weight || 'normal').toLowerCase();
  const s = String(style || 'normal').toLowerCase();

  let base;
  if (w === 'bold' || w === '700') base = 'Bold';
  else if (w === '600' || w === 'semibold') base = 'Semi Bold';
  else if (w === '500' || w === 'medium') base = 'Medium';
  else base = 'Regular';

  if (s === 'italic') {
    return base === 'Regular' ? 'Italic' : base + ' Italic';
  }
  return base;
}

// ─── Main ───────────────────────────────────────────────────────────

async function createDiagram(spec) {
  figma.ui.postMessage({ type: 'progress', message: 'Loading fonts...' });

  // Collect all required font styles
  const styles = new Set(['Regular', 'Bold', 'Semi Bold', 'Italic', 'Bold Italic']);
  for (const el of spec.elements) {
    if (el.type === 'text') {
      styles.add(el.figmaStyle || resolveFontStyle(el.fontWeight, el.fontStyle));
    }
  }

  // Load fonts
  for (const s of styles) {
    try {
      await figma.loadFontAsync(figmaFontName(s));
    } catch (e) {
      console.warn(`Font Inter ${s} not available, skipping`);
    }
  }

  figma.ui.postMessage({ type: 'progress', message: 'Creating elements...' });

  // Master frame
  const frame = figma.createFrame();
  frame.name = 'CTCF Architecture';
  frame.resize(spec.canvas.width, spec.canvas.height);
  frame.fills = solidPaint(spec.canvas.background || '#FAFAFA');
  frame.clipsContent = false;

  // Track group membership: groupId -> [node]
  const groupNodes = {};
  const groupMeta = {};
  for (const g of (spec.groups || [])) {
    groupMeta[g.id] = g;
    groupNodes[g.id] = [];
  }

  let count = 0;

  for (const el of spec.elements) {
    let node = null;

    switch (el.type) {
      case 'rect':
        node = makeRect(el);
        break;
      case 'text':
        node = makeText(el);
        break;
      case 'arrow':
        node = makeArrow(el);
        break;
    }

    if (!node) continue;

    // Name the node
    if (el.name) {
      node.name = el.name;
    } else if (el.type === 'text') {
      const preview = (el.text || '').substring(0, 30);
      node.name = `Text: ${preview}`;
    }

    // Add to frame
    frame.appendChild(node);

    // Track group
    if (el.group && groupNodes[el.group]) {
      groupNodes[el.group].push(node);
    }

    count++;

    // Handle subscript: create a second smaller text node
    if (el.type === 'text' && el.sub) {
      const subNode = makeSubscript(el, node);
      if (subNode) {
        frame.appendChild(subNode);
        if (el.group && groupNodes[el.group]) {
          groupNodes[el.group].push(subNode);
        }
        count++;
      }
    }
  }

  // Create groups (bottom-up: leaf groups first)
  figma.ui.postMessage({ type: 'progress', message: 'Grouping elements...' });
  const sortedGroups = (spec.groups || []).slice().reverse();
  for (const g of sortedGroups) {
    const children = groupNodes[g.id];
    if (children && children.length > 1) {
      try {
        const group = figma.group(children, frame);
        group.name = g.name || g.id;
        // If this group belongs to a parent group, add it there
        if (g.parent && groupNodes[g.parent]) {
          groupNodes[g.parent].push(group);
        }
      } catch (e) {
        console.warn(`Could not group ${g.id}: ${e.message}`);
      }
    }
  }

  // Zoom to fit
  figma.viewport.scrollAndZoomIntoView([frame]);

  return count;
}

// ─── Element creators ───────────────────────────────────────────────

function makeRect(el) {
  const rect = figma.createRectangle();
  rect.x = el.x;
  rect.y = el.y;
  rect.resize(Math.max(1, el.w), Math.max(1, el.h));
  rect.cornerRadius = el.cornerRadius || 0;

  if (el.fill && el.fill !== 'none') {
    rect.fills = solidPaint(el.fill);
  } else {
    rect.fills = [];
  }

  if (el.stroke && el.stroke !== 'none') {
    rect.strokes = solidPaint(el.stroke);
    rect.strokeWeight = el.strokeWeight || 1;
    rect.strokeAlign = 'INSIDE';
  }

  if (el.dash && el.dash.length > 0) {
    rect.dashPattern = el.dash;
  }

  if (el.opacity !== undefined && el.opacity !== 1) {
    rect.opacity = el.opacity;
  }

  return rect;
}

function makeText(el) {
  const text = figma.createText();
  const style = el.figmaStyle || resolveFontStyle(el.fontWeight, el.fontStyle);

  try {
    text.fontName = figmaFontName(style);
  } catch (e) {
    text.fontName = figmaFontName('Regular');
  }

  text.characters = el.text || '';
  text.fontSize = el.fontSize || 13;
  text.textAutoResize = 'WIDTH_AND_HEIGHT';

  if (el.fill) {
    const fills = solidPaint(el.fill);
    if (fills.length > 0) text.fills = fills;
  }

  // Position: convert from SVG anchor/baseline to Figma top-left
  const anchor = el.anchor || 'center';
  const baselineOffset = text.height * 0.75;

  if (anchor === 'center' || anchor === 'middle') {
    text.x = el.x - text.width / 2;
  } else if (anchor === 'end' || anchor === 'right') {
    text.x = el.x - text.width;
  } else {
    text.x = el.x;  // start/left
  }

  text.y = el.y - baselineOffset;

  return text;
}

function makeSubscript(el, mainNode) {
  // Create a smaller text node positioned as subscript
  const sub = figma.createText();
  const style = resolveFontStyle(el.fontWeight, el.fontStyle);

  try {
    sub.fontName = figmaFontName(style);
  } catch (e) {
    sub.fontName = figmaFontName('Regular');
  }

  sub.characters = el.sub;
  sub.fontSize = Math.max(7, (el.fontSize || 13) - 3);
  sub.textAutoResize = 'WIDTH_AND_HEIGHT';

  if (el.fill) {
    sub.fills = solidPaint(el.fill);
  }

  // Position: to the right and slightly below the main text
  sub.x = mainNode.x + mainNode.width + 1;
  sub.y = mainNode.y + mainNode.height * 0.45;
  sub.name = `Sub: ${el.sub}`;

  return sub;
}

function makeArrow(el) {
  const points = el.points;
  if (!points || points.length < 2) return null;

  // Compute bounding box
  let minX = Infinity, minY = Infinity;
  for (const [px, py] of points) {
    minX = Math.min(minX, px);
    minY = Math.min(minY, py);
  }

  // Build vertices in local coordinates
  const vertices = points.map(([px, py], i) => {
    let cap = 'NONE';
    if (i === points.length - 1 && el.endArrow !== false) {
      cap = 'ARROW_LINES';
    }
    if (i === 0 && el.startArrow) {
      cap = 'ARROW_LINES';
    }
    return {
      x: px - minX,
      y: py - minY,
      strokeCap: cap,
      cornerRadius: 0,
    };
  });

  // Build segments
  const segments = [];
  for (let i = 0; i < vertices.length - 1; i++) {
    segments.push({ start: i, end: i + 1 });
  }

  const vec = figma.createVector();
  vec.x = minX;
  vec.y = minY;

  try {
    vec.vectorNetwork = { vertices, segments, regions: [] };
  } catch (e) {
    // Fallback: create a simple line
    console.warn('vectorNetwork failed:', e.message);
    const dx = points[points.length - 1][0] - points[0][0];
    const dy = points[points.length - 1][1] - points[0][1];
    vec.resize(Math.sqrt(dx * dx + dy * dy) || 1, 0);
    vec.rotation = -Math.atan2(dy, dx) * 180 / Math.PI;
  }

  vec.strokes = solidPaint(el.stroke || '#2C3E50');
  vec.strokeWeight = el.strokeWeight || 1.5;
  vec.fills = [];

  if (el.dash && el.dash.length > 0) {
    vec.dashPattern = el.dash;
  }

  return vec;
}
