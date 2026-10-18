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

uniform sampler2D tLocH;   // RG32F  bed, water delta (2 m, map-aligned)
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
uniform sampler2D tCSd;    // R8     castle 2D signed distance (0.5 m)
uniform sampler2D tCN;     // RGBA8  castle top, roof type, tower idx 1, tower idx 2 (nearest)
uniform sampler2D tFor;    // RG8    essay Forest-map density, field fraction (8 m)

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
vec2 texRG(sampler2D s, vec2 uv) {
#ifdef MANUAL_BILERP
  ivec2 sz = textureSize(s, 0);
  vec2 x = uv * vec2(sz) - 0.5; vec2 i = floor(x); vec2 f = x - i;
  ivec2 a = clamp(ivec2(i), ivec2(0), sz - 1), b = clamp(ivec2(i) + 1, ivec2(0), sz - 1);
  vec2 v00 = texelFetch(s, a, 0).rg, v10 = texelFetch(s, ivec2(b.x, a.y), 0).rg;
  vec2 v01 = texelFetch(s, ivec2(a.x, b.y), 0).rg, v11 = texelFetch(s, b, 0).rg;
  return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
#else
  return textureLod(s, uv, 0.).rg;
#endif
}
vec2 uvM(vec2 xz) { return (xz + 5005.) / 10010.; }
vec2 uvA(vec2 xz) { return (xz + 12015.) / 24030.; }
vec2 uvB(vec2 xz) { return (xz + 36060.) / 72120.; }

vec2 regionalH(vec2 xz) {
  float m = max(abs(xz.x), abs(xz.y));
  if (m < 4700.) return texRG(tMH, uvM(xz));
  if (m < 11600.) {
    vec2 a = texRG(tAH, uvA(xz));
    if (m < 4990.) return mix(texRG(tMH, uvM(xz)), a, smoothstep(4700., 4990., m));
    return a;
  }
  vec2 b = texRG(tBH, uvB(xz));
  if (m < 11990.) return mix(texRG(tAH, uvA(xz)), b, smoothstep(11600., 11990., m));
  return b;
}

struct GS { float bed; float wd; float lw; float ed; vec2 luv; vec4 la; };

GS sampleGround(vec2 xz) {
  GS g;
  vec2 r = regionalH(xz);
  g.luv = vec2(dot(uLocU, vec3(xz, 1.)), dot(uLocV, vec3(xz, 1.)));
  g.lw = 0.; g.la = vec4(0., 0., 1., 1.);
  vec2 e = min(g.luv, 1. - g.luv) * uLocM;
  float ed = min(e.x, e.y);
  g.ed = ed;
  if (ed > 0.) {
    g.lw = smoothstep(0., 80., ed);
    r = mix(r, texRG(tLocH, g.luv), g.lw);
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
  if (fw > 0.) d = textureLod(tFor, fuv, 0.).r * fw;
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

// per-tree canopy heightfield: 7 m jittered cells, 2x2 neighbourhood
float treeTop(vec2 xz, float gy, float dens, float conF, out vec4 tc, out vec4 tp) {
  vec2 q = xz / 7.; vec2 c0 = floor(q); vec2 f = q - c0;
  vec2 o = vec2(f.x < 0.5 ? -1. : 1., f.y < 0.5 ? -1. : 1.);
  float best = -1e9; tc = vec4(0.); tp = vec4(0.);
  for (int k = ZERO; k < 4; k++) {
    vec2 cc = c0 + vec2(float(k & 1), float(k >> 1)) * o;
    vec4 h = hash42(cc);
    if (h.x >= dens) continue;
    vec2 ctr = (cc + 0.25 + 0.5 * h.yz) * 7.;
    float hh = fract(h.w * 13.7);
    bool con = h.w < conF;
    float H = (con ? mix(13., 21., hh) : mix(10., 19., hh)) * (0.72 + 0.28 * smoothstep(0.1, 0.8, dens));
    float r = con ? H * 0.23 : mix(3.0, 4.9, fract(hh * 7.3));
    float dd = length(xz - ctr);
    if (dd >= r) continue;
    float x = dd / r, a = 0., top;
    if (con) top = gy + H * (1. - x);
    else { a = 0.42 * H; top = gy + H - a * (1. - sqrt(1. - x * x)); }
    if (top > best) { best = top; tc = vec4(ctr, r, H); tp = vec4(con ? 1. : 0., a, h.w, 1.); }
  }
  return best;
}

float vegTop(vec2 xz, float gy, float dens, float conF, float t, out vec4 tc, out vec4 tp) {
  tc = vec4(0.); tp = vec4(0.);
  if (dens < 0.03) return -1e9;
  if (t < TREE_FAR) return treeTop(xz, gy, dens, conF, tc, tp);
  tp = vec4(conF > 0.5 ? 1. : 0., 0., 0.5, -1.);     // far canopy as a smooth sheet
  return gy + 14. * smoothstep(0.05, 0.55, dens) * (0.8 + 0.4 * vnoise(xz * 0.08));
}
