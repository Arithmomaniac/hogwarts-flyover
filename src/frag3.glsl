// ------------------------------------------------------------------ combined terrain / water / vegetation evaluation
struct TS { GS g; float gy; float wy; float vy; float vd; float dens; float conF; vec4 tc; vec4 tp; };

// vm: 0 no vegetation, 1 trees (3D trees near, canopy sheet far), 2 canopy sheet only
// vy: canopy-sheet height (folded into the terrain height); vd: distance bound to the 3D trees
TS evalTop(vec3 p, float td, int vm) {
  TS s;
  s.g = sampleGround(p.xz);
  s.gy = s.g.bed + detailH(p.xz, s.g, td);
  s.wy = s.g.bed + s.g.wd;
  s.vy = -1e9; s.vd = 1e5; s.dens = 0.; s.conF = 0.; s.tc = vec4(0.); s.tp = vec4(0.);
  if (vm > 0 && (uDbg & 8) == 0) {
    float d0 = vegDensity(p.xz, s.g, s.conF);
    s.dens = d0 * vegMask(s.g);
    float ty = max(s.gy, s.wy);
#ifndef FAR_TILE
    // FAR_TILE: drawn only on blocks whose cone start is >= TREE_FAR, so no sample there reaches this branch
    if (vm == 1 && td < TREE_FAR) {                  // 3D trees, wherever the tree table has one within two cells
      vec2 t0 = floor(p.xz / 7.) + TREE_G0;
      bool tab = min(t0.x, t0.y) >= 1. && max(t0.x, t0.y) < float(TREE_N) - 1.;
      if (p.y - ty >= VEGMAX + 1.) { if (tab || d0 >= 0.03) s.vd = p.y - ty - VEGMAX; }   // above every tree: a bound only
      else {
        float mh = tab ? texelFetch(tTree, ivec2(t0), 0).b * 25.5 : (d0 >= 0.03 ? VEGMAX : 0.);
        if (mh > 0.5) s.vd = treeSD(p, s.gy, mh, tab, s.tc, s.tp);
        else if (tab) s.vd = 10.85;
      }
    } else
#endif
    if (s.dens >= 0.03)
      s.vy = p.y >= ty + VEGMAX + 1. ? ty + VEGMAX : canopyTop(p.xz, s.gy, s.dens, s.conF, s.tc, s.tp);
  }
  return s;
}
float topOf(TS s) { return max(max(s.gy, s.wy), s.vy); }

vec3 vegNormal(vec3 p, TS s, vec3 gn) {
  if (s.tp.w < 0.) return normalize(mix(gn, vec3(0., 1., 0.), 0.55));
  vec2 d = p.xz - s.tc.xy;
  vec3 n;
  if (s.tp.w > 1.5) return normalize(vec3(d.x, 0., d.y) + 1e-4);   // trunk
  if (s.tp.x > 0.5) {
    vec2 u = normalize(d + 1e-4) * (0.88 * s.tc.w / s.tc.z);
    n = normalize(vec3(u.x, 1., u.y));
  } else {
    float a = max(s.tp.y, 0.5);
    float cy = s.gy + s.tc.w - a;
    n = normalize(vec3(d.x / (s.tc.z * s.tc.z), (p.y - cy) / (a * a), d.y / (s.tc.z * s.tc.z)));
  }
  vec3 j = vec3(vnoise(p.xz * 1.9 + p.y), vnoise(p.zy * 1.9 + 4.), vnoise(p.xy * 1.9 + 8.)) - 0.5;
  return normalize(n + j * 0.9);
}

// ------------------------------------------------------------------ one loop for every ray query
// The scene is evaluated at exactly one call site; the D3D shader compiler behind WebGL on Windows
// otherwise inlines it once per use (march, refine, normals, reflection) and takes minutes.
#define ST_MARCH 0
#define ST_BISECT 1
#define ST_GNORM 3
#define ST_SHINIT 4
#define ST_RINIT 6
#define ST_REFL 7
#define ST_RNORM 8
#define ST_DONE 9

// mat: 0 sky, 1 ground, 2 water, 3 vegetation; 4 castle, 5 building, 6 wall/hedge, 7 set piece come from the mesh pass
// reflection rmat: 0 sky, 1 ground, 3 canopy; 5 a mirrored mesh (the shading pass)
struct Res {
  float t; int mat; float aux; float ctop; TS s; vec3 n; vec3 wn; float sh; float it;
  vec3 rd; int rmat; float rt; vec3 rp; vec3 rn; float rconF;
#ifdef SPLIT_P1
  vec3 so; float stt; bool shf;
#endif
};

// far-field sun visibility, baked per 7 m tree cell into tTree.a by SUN_PASS (bilinear over cell centres)
#define SUN_NEAR 3200.0
#define SUN_TREF 5000.0
// the ground normal's finite-difference step (m): wider within the water layer's reach, where a bank drops through one
// 2 m texel and the bilinear cells would otherwise shade as a staircase along the shore
#define GNORM_E(t, s) max(clamp((t) * 0.0018, 0.3, 45.), (s).g.wd > -0.9 ? 2. : 0.)
#ifdef SPLIT_P1
float sunCache(vec2 xz) {
  vec2 x = xz / 7. + TREE_G0 - 0.5, i = floor(x), f = x - i;
  ivec2 a = clamp(ivec2(i), ivec2(0), ivec2(TREE_N - 1)), b = clamp(ivec2(i) + 1, ivec2(0), ivec2(TREE_N - 1));
  float v00 = texelFetch(tTree, a, 0).a, v10 = texelFetch(tTree, ivec2(b.x, a.y), 0).a;
  float v01 = texelFetch(tTree, ivec2(a.x, b.y), 0).a, v11 = texelFetch(tTree, b, 0).a;
  return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
}

// The primary march (P1 and FAR_TILE): terrain, water and trees, and each hit's lake reflection. The shadow pass
// marches the sun term from the start trace() records; the meshes draw themselves over this record where nearer.
Res trace(vec3 ro, vec3 rd) {
  Res r;
  r.t = TMAX; r.mat = 0; r.aux = 0.; r.ctop = 0.; r.n = vec3(0., 1., 0.); r.wn = r.n; r.sh = 1.; r.it = 0.;
  r.rd = rd; r.rmat = 0; r.rt = 0.; r.rp = vec3(0.); r.rn = r.n; r.rconF = 0.;
  r.so = vec3(0.); r.stt = 0.; r.shf = false;
  int maxM = uQual == 0 ? 170 : (uQual == 1 ? 250 : 360);
  int maxR = (uQual == 0 || uDetail == 0 || (uDbg & 2) != 0) ? 0 : (uQual == 1 ? 64 : 100);
  int st = ST_MARCH, cnt = 0;
  float t = 0.05, tPrev = 0.05, ta = 0., tb = 0., e = 0., g0 = 0., h1 = 0., rtt = 0.;
  float tMesh = texelFetch(tMeshT, ivec2(gl_FragCoord.xy), 0).r;
  vec3 hp = vec3(0.), rro = vec3(0.);
  if (ro.y > CEIL) { if (rd.y >= 0.) st = ST_DONE; else t = (ro.y - CEIL) / -rd.y; }
  if (uCone != 0 && st == ST_MARCH) {               // resume the baseline march state the block's cone pass replayed
    uvec4 cb = uvec4(texelFetch(tTree, ivec2(gl_FragCoord.xy) / 8 + ivec2(0, TREE_N), 0) * 255. + 0.5);
    uint u = cb.r | (cb.g << 8) | (cb.b << 16) | (cb.a << 24);
    float tc = uintBitsToFloat(u & ~511u);          // start t_k (low bits truncated: never later than t_k)
    if (u != 0u && tc > t && tc <= TMAX) { t = tc; tPrev = tc; cnt = int(u & 511u); }   // skipped steps stay charged to maxM
  }
  int nIt = maxM + maxR + 16;
  for (int i = ZERO; i < nIt; i++) {
    // ---- transitions that need no scene sample
    if (st == ST_SHINIT) {                           // where the shadow pass starts the sun march, or the sun term itself
      vec2 dd = hp.xz - r.s.tc.xy;                   // tree hits: start outward from the crown, not straight up
      vec3 vo = r.s.tp.w > 0. ? normalize(vec3(dd.x, max(length(dd), 0.5) * 0.6, dd.y)) : vec3(0., 0.6, 0.);
      r.so = hp + (r.mat == 3 ? vo : (r.mat == 2 ? vec3(0., 0.1, 0.) : r.n * (0.04 + r.t * 0.0008))) + uSunDir * 0.05;
      r.stt = 0.12 + r.t * 0.0015;
      r.sh = uSunDir.y < -0.03 ? 0. : 1.;
      r.shf = uSunDir.y > -0.03 && (uDbg & 1) == 0;
      if (r.shf && r.t >= SUN_NEAR && max(abs(hp.x), abs(hp.z)) < 5040.) { r.sh = sunCache(hp.xz); r.shf = false; }
      st = ST_RINIT;
    }
    if (st == ST_RINIT) {
      st = ST_DONE;
      if (r.mat == 2) {
        r.wn = waterNormal(hp.xz, r.t, r.t * 2. * uTanHalf / uRes.y);
        vec3 rr = reflect(rd, r.wn); rr.y = max(rr.y, 0.003); r.rd = normalize(rr);
        if (maxR > 0) { rro = hp + vec3(0., 0.08, 0.); rtt = 0.6; cnt = 0; st = ST_REFL; }
      }
    }
    if (st == ST_DONE) break;
    r.it += 1.;
    // ---- where to sample and what the sample needs (vm: vegetation mode)
    vec3 p = vec3(0.); float td = 0.; int vm = 0;
    if (st == ST_MARCH) { p = ro + rd * t; td = t; vm = uDetail == 1 ? 1 : 2; }
    else if (st == ST_BISECT) { t = 0.5 * (ta + tb); p = ro + rd * t; td = t; vm = uDetail == 0 ? 2 : (t < TREE_FAR ? 0 : 1); }
    else if (st == ST_GNORM || st == ST_RNORM) { p = hp + (cnt == 0 ? vec3(e, 0., 0.) : vec3(0., 0., e)); td = st == ST_GNORM ? r.t : 1e5; }
    else { p = rro + r.rd * rtt; td = 1e5; vm = 2; }
    // ---- the single scene evaluation
    TS s = evalTop(p, td, vm);
    float dE = s.g.ed > 0. ? 1e5 : max(1. - s.g.ed, 1.);   // the local grid's edge: it bounds the step, never a hit
    float dy = p.y - topOf(s);
    // ---- advance
    if (st == ST_MARCH) {
      float eps = min(0.00045 * t + 0.01, 0.9);
      if (t > TMAX || (p.y > CEIL && rd.y > 0.)) st = ST_DONE;
      else if (s.vd < eps) {                         // 3D tree: shading takes its normal from the tree's shape
        r.t = t; r.mat = 3; r.s = s; hp = p; cnt = 0; st = ST_SHINIT;
      } else if (dy < 0.) { ta = tPrev; tb = t; cnt = 0; st = ST_BISECT; }
      // a mesh is nearer than anything found so far: its own pass fills the pixel. The step bounds cannot pass over a
      // tree, and terrain crossed on the way is bisected above, so nothing in front of the mesh is missed
      else if (t >= tMesh) st = ST_DONE;
      else if (++cnt >= maxM) {                      // out of steps: grazing ray, call it ground
        r.t = t; r.mat = 1; r.s = s; hp = p; g0 = s.gy; e = GNORM_E(t, s); cnt = 0; st = ST_GNORM;
      } else {
        tPrev = t;
        float k = mix(0.55, 0.85, smoothstep(900., 7000., t));
        t += min(min(max(k * dy, 0.0016 * t + 0.03), max(dE, eps)), max(0.8 * s.vd, max(eps, 0.0012 * t)));   // 0.8: trees ride the terrain slope
      }
    } else if (st == ST_BISECT) {
      if (dy < 0.) tb = t; else ta = t;
      if (++cnt >= 7) {
        float ty = max(s.gy, s.wy);
        r.t = t; r.s = s; hp = p; g0 = s.gy;
        r.mat = (s.vy > ty && s.tp.w != 0.) ? 3 : (s.wy >= s.gy ? 2 : 1);
        e = GNORM_E(t, s); cnt = 0; st = ST_GNORM;
      }
    } else if (st == ST_GNORM || st == ST_RNORM) {
      if (cnt == 0) h1 = s.gy;
      else {
        vec3 n = normalize(vec3(g0 - h1, e, g0 - s.gy));
        if (st == ST_GNORM) { r.n = n; st = ST_SHINIT; } else { r.rn = n; st = ST_DONE; }
      }
      cnt++;
    } else {                                         // ST_REFL
      if (dy < 0.) {
        r.rt = rtt; r.rp = p; r.rconF = s.conF;
        r.rmat = s.vy > max(s.gy, s.wy) ? 3 : 1;
        hp = p; g0 = s.gy; e = 8. + rtt * 0.004; cnt = 0; st = ST_RNORM;
      } else {
        rtt += clamp(min(0.6 * dy, dE), 0.5 + rtt * 0.015, 300.);
        if (p.y > CEIL || rtt > 9000. || ++cnt >= maxR) st = ST_DONE;
      }
    }
  }
  return r;
}
#endif
