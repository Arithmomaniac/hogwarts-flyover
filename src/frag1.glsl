#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
//__DEFINES__
out vec4 outColor;

uniform vec2 uRes;
uniform vec2 uJitter;      // sub-pixel offset for temporal upsampling (scene pixels)
uniform float uTime;
uniform vec3 uCamPos;
uniform mat3 uCamRot;
uniform float uTanHalf;
uniform vec3 uSunDir;
uniform vec3 uSunCol;
uniform vec3 uSkyCol;
uniform vec3 uHorCol;
uniform float uDusk;
uniform vec3 uGlow;       // sunset/twilight glow toward the sun's azimuth
uniform float uExposure;
uniform int uQual;
uniform int uDetail;       // 1 full detail, 0 lite (canopy sheet instead of trees, no lake reflections)
uniform int uDbg;

uniform sampler2D tLocH;   // RGB32F bed, water delta, castle base height (2 m, map-aligned)
uniform sampler2D tLocA;   // RGBA8  road, trees, hedge sd, building sd
uniform sampler2D tLocB;   // RG8    boundary-wall sd, stream sd
uniform sampler2D tLocN;   // RG8    surface class, building code (nearest)
uniform sampler2D tMH;     // RG32F  10 m  (+-5 km)
uniform sampler2D tMS;     // sRGB   Sentinel-2 albedo
uniform sampler2D tMF;     // R8     plantation density
uniform sampler2D tAH;     // 30 m (+-12 km)
uniform sampler2D tAS;
uniform sampler2D tAF;
uniform sampler2D tBH;     // 120 m (+-36 km)
uniform sampler2D tBS;
uniform sampler2D tCasA;   // RGBA8  castle footprint sd of mass classes 0-3 (0.5 m)
uniform sampler2D tCasB;   // RGBA8  castle footprint sd of mass classes 4-5; nearest two tower indices (texelFetch)
uniform sampler2D tFor;    // RG8    essay Forest-map density, field fraction (8 m)
uniform sampler2D tTree;   // RGBA8  per 7 m tree cell (cells -TREE_G0..): 3D-tree density, conifer fraction, tallest tree within two cells / 25.5 m (rendered at load)

uniform vec3 uLocU; uniform vec3 uLocV; uniform vec2 uLocM;
uniform vec3 uCasU; uniform vec3 uCasV;
uniform vec3 uForU; uniform vec3 uForV; uniform vec2 uForM;
uniform vec4 uTw[NTW];     // tower: east, north, radius, top
uniform vec4 uTx[NTW];     // tower: roof type

// non-constant zero: stops the D3D compiler from unrolling loops (and inlining their bodies N times)
#define ZERO min(uQual, 0)

#define CEIL 1240.0
#define TMAX 46000.0
#define VEGMAX 23.0
#define TREE_FAR 2600.0
#define PI 3.14159265

// ------------------------------------------------------------------ hashing / noise
float hash12(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
vec4 hash42(vec2 p) { vec4 p4 = fract(vec4(p.xyxy) * vec4(0.1031, 0.1030, 0.0973, 0.1099)); p4 += dot(p4, p4.wzxy + 33.33); return fract((p4.xxyz + p4.yzzw) * p4.zywx); }
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3. - 2. * f);
  float a = hash12(i), b = hash12(i + vec2(1, 0)), c = hash12(i + vec2(0, 1)), d = hash12(i + vec2(1, 1));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbm4(vec2 p) { float s = 0., a = 0.5; for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = mat2(1.6, 1.2, -1.2, 1.6) * p; a *= 0.5; } return s; }

// loch surface ripples; octaves fade out once they fall below the pixel footprint (pix, metres)
float waveH(vec2 xz, float pix) {
  vec2 w = uTime * vec2(0.6, 0.25);
  return 0.5 * vnoise(xz * 0.09 + w * 0.05)
       + 0.3 * vnoise(xz * 0.31 - w * 0.12) * (1. - smoothstep(4., 12., pix))
       + 0.2 * vnoise(xz * 1.05 + w * 0.4) * (1. - smoothstep(1.2, 3.5, pix));
}
vec3 waterNormal(vec2 xz, float t, float pix) {
  float e = 0.25;
  float h0 = waveH(xz, pix), hx = waveH(xz + vec2(e, 0.), pix), hz = waveH(xz + vec2(0., e), pix);
  float s = 0.15 * (1. - smoothstep(150., 2500., t)) + 0.025;
  return normalize(vec3((h0 - hx) / e * s, 1., (h0 - hz) / e * s));
}

float sdBox(vec3 p, vec3 b) { vec3 q = abs(p) - b; return length(max(q, 0.)) + min(max(q.x, max(q.y, q.z)), 0.); }
float sdRBox(vec3 p, vec3 b, float r) { vec3 q = abs(p) - b + r; return length(max(q, 0.)) + min(max(q.x, max(q.y, q.z)), 0.) - r; }
float sdEll(vec3 p, vec3 r) { float k0 = length(p / r); float k1 = length(p / (r * r)); return k0 * (k0 - 1.) / max(k1, 1e-6); }
float sdCone(vec2 w, float R, float h) { return max(dot(w - vec2(R, 0.), normalize(vec2(h, R))), -w.y); }   // w = (radial, height above base)

// ------------------------------------------------------------------ height textures
vec3 texRG(sampler2D s, vec2 uv) {
#ifdef MANUAL_BILERP
  ivec2 sz = textureSize(s, 0);
  vec2 x = uv * vec2(sz) - 0.5; vec2 i = floor(x); vec2 f = x - i;
  ivec2 a = clamp(ivec2(i), ivec2(0), sz - 1), b = clamp(ivec2(i) + 1, ivec2(0), sz - 1);
  vec3 v00 = texelFetch(s, a, 0).rgb, v10 = texelFetch(s, ivec2(b.x, a.y), 0).rgb;
  vec3 v01 = texelFetch(s, ivec2(a.x, b.y), 0).rgb, v11 = texelFetch(s, b, 0).rgb;
  return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
#else
  return textureLod(s, uv, 0.).rgb;
#endif
}
vec2 uvM(vec2 xz) { return (xz + 5005.) / 10010.; }
vec2 uvA(vec2 xz) { return (xz + 12015.) / 24030.; }
vec2 uvB(vec2 xz) { return (xz + 36060.) / 72120.; }

vec2 regionalH(vec2 xz) {
  float m = max(abs(xz.x), abs(xz.y));
  if (m < 4700.) return texRG(tMH, uvM(xz)).xy;
  if (m < 11600.) {
    vec2 a = texRG(tAH, uvA(xz)).xy;
    if (m < 4990.) return mix(texRG(tMH, uvM(xz)).xy, a, smoothstep(4700., 4990., m));
    return a;
  }
  vec2 b = texRG(tBH, uvB(xz)).xy;
  if (m < 11990.) return mix(texRG(tAH, uvA(xz)).xy, b, smoothstep(11600., 11990., m));
  return b;
}

struct GS { float bed; float wd; float lw; float ed; vec2 luv; vec4 la; float cb; };

GS sampleGround(vec2 xz) {
  GS g;
  vec2 r = regionalH(xz);
  g.luv = vec2(dot(uLocU, vec3(xz, 1.)), dot(uLocV, vec3(xz, 1.)));
  g.lw = 0.; g.la = vec4(0., 0., 1., 1.); g.cb = 0.;
  vec2 e = min(g.luv, 1. - g.luv) * uLocM;
  float ed = min(e.x, e.y);
  g.ed = ed;
  if (ed > 0.) {
    g.lw = smoothstep(0., 80., ed);
    vec3 l = texRG(tLocH, g.luv);
    r = mix(r, l.xy, g.lw); g.cb = l.z;
    g.la = textureLod(tLocA, g.luv, 0.);
  }
  g.bed = r.x; g.wd = r.y;
  return g;
}

float bldSd(GS g) { return mix(12.7, g.la.w * 25.5 - 12.8, step(0.01, g.lw)); }

// procedural micro-relief on top of the ~30 m Copernicus DEM
float detailH(vec2 xz, GS g, float t) {
  if (t > 3200.) return 0.;
  float amp = mix(1.7, 0.2, g.lw);
  amp *= clamp(-g.wd * 0.3, 0., 1.);
  amp *= (1. - g.la.r * g.lw) * smoothstep(0., 3., bldSd(g));
  float n = (vnoise(xz * 0.043) - 0.5) * amp + (vnoise(xz * 0.161 + 17.) - 0.5) * amp * 0.4;
  if (t < 700.) n += (vnoise(xz * 0.57 + 3.) - 0.5) * amp * 0.15;
  return n * (1. - smoothstep(2200., 3200., t));
}

// ------------------------------------------------------------------ vegetation
float vegDensity(vec2 xz, GS g, out float conF) {
  conF = 0.12;                                       // essay: mainly deciduous, admixture of conifers
  float d = 0.;
  vec2 fuv = vec2(dot(uForU, vec3(xz, 1.)), dot(uForV, vec3(xz, 1.)));
  vec2 fe = min(fuv, 1. - fuv) * uForM;
  float fw = smoothstep(0., 300., min(fe.x, fe.y));
  if (fw > 0. && g.lw < 1.) d = textureLod(tFor, fuv, 0.).r * fw;
  if (g.lw > 0.) d = mix(d, g.la.g, g.lw);
  float rr = length(xz);
  float sw = smoothstep(2500., 3500., rr);
  if (sw > 0.) {                                     // real conifer plantations (Sentinel-2), outside the essay's core
    float m = max(abs(xz.x), abs(xz.y));
    float s = m < 4950. ? textureLod(tMF, uvM(xz), 0.).r : (m < 11950. ? textureLod(tAF, uvA(xz), 0.).r : 0.);
    s *= sw;
    if (s > d) { conF = mix(conF, 1., smoothstep(0., 0.25, s - d)); d = s; }
  }
  conF = max(conF, 1. - smoothstep(120., 230., length(xz - STATION)));   // pine bank below the station
  conF = max(conF, 0.85 * (1. - smoothstep(260., 480., length(xz - GRAWP))));  // pines round Grawp's clearing
  return d;
}

float vegMask(GS g) { return step(g.wd, -0.9) * (1. - g.la.r * g.lw) * smoothstep(0.5, 2.5, bldSd(g)); }

// density and conifer fraction at a tree's own position, so each tree is the same whichever ray finds it
float treeDens(vec2 xz, out float conF) {
  GS g;
  g.luv = vec2(dot(uLocU, vec3(xz, 1.)), dot(uLocV, vec3(xz, 1.)));
  vec2 e = min(g.luv, 1. - g.luv) * uLocM;
  g.ed = min(e.x, e.y);
  if (g.ed > 80.) {                                  // well inside the local layer only its textures matter
    g.lw = 1.; g.la = textureLod(tLocA, g.luv, 0.); g.wd = texRG(tLocH, g.luv).y; g.bed = 0.; g.cb = 0.;
  } else g = sampleGround(xz);
  return vegDensity(xz, g, conF) * vegMask(g);
}

// height of the tree in a cell with hash h (r: crown radius)
float treeH(vec4 h, float dens, bool con, out float r) {
  float hh = fract(h.w * 13.7);
  float H = (con ? mix(13., 21., hh) : mix(10., 19., hh)) * (0.72 + 0.28 * smoothstep(0.1, 0.8, dens));
  r = con ? H * 0.23 : mix(3.0, 4.9, fract(hh * 7.3));
  return H;
}

// per-tree 3D shapes (crown on a trunk) within TREE_FAR: 7 m jittered cells, 3x3 neighbourhood.
// Returns a distance bound that also covers trees in cells not visited (TREE_SAFE away) and, when a crown or trunk
// is the nearest thing (tp.w > 0), that tree: tc = (centre, crown radius, height), tp = (conifer, crown half-height,
// hue, 1 crown / 2 trunk). mh: tallest tree within two cells (tab: from tTree); trees beyond are >= 10.85 m away.
#define TREE_SAFE 3.85
float treeSD(vec3 p, float gy, float mh, bool tab, out vec4 tc, out vec4 tp) {
  vec2 c0 = floor(p.xz / 7.);
  float best = TREE_SAFE, hy = p.y - gy;
  tc = vec4(0.); tp = vec4(0.);
  if (hy - mh > 1.) return min(hy - mh, 10.85);     // clear above every crown nearby (> 1 m: never a hit)
  for (int k = ZERO; k < 9; k++) {
    int j = (k + 4) % 9;                             // own cell first: tightens the bound sooner
    vec2 cc = c0 + vec2(float(j % 3), float(j / 3)) - 1.;
    vec4 h = hash42(cc);
    vec2 ctr = (cc + 0.25 + 0.5 * h.yz) * 7., q = p.xz - ctr;
    float rq = length(q);
    if (rq - 4.9 > best) continue;                   // no crown from this cell can be nearer
    float conF, dens;
    if (tab) { vec2 v = texelFetch(tTree, ivec2(cc + TREE_G0), 0).rg; dens = v.x; conF = v.y; }
    else dens = treeDens(ctr, conF);
    if (h.x >= dens) continue;
    bool con = h.w < conF;
    float r, H = treeH(h, dens, con, r);
    if (rq - r > best) continue;
    float a = con ? 0. : 0.36 * H, dc;
    if (con) dc = sdCone(vec2(rq, hy - 0.12 * H), r, 0.88 * H);
    else dc = sdEll(vec3(q.x, hy - (H - a), q.y), vec3(r, a, r));
    float dt = max(rq - 0.25 - 0.012 * H, max(hy - (con ? 0.5 * H : H - a), -hy - 0.5));
    float dd = min(dc, dt);
    if (dd < best) { best = dd; tc = vec4(ctr, r, H); tp = vec4(con ? 1. : 0., a, h.w, dc <= dt ? 1. : 2.); }
  }
  return best;
}

// far canopy as a smooth sheet (beyond TREE_FAR, lite detail and lake reflections)
float canopyTop(vec2 xz, float gy, float dens, float conF, out vec4 tc, out vec4 tp) {
  tc = vec4(0.);
  tp = vec4(conF > 0.5 ? 1. : 0., 0., 0.5, -1.);
  return gy + 14. * smoothstep(0.05, 0.55, dens) * (0.8 + 0.4 * vnoise(xz * 0.08));
}
