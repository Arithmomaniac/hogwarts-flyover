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
    if (vm == 1 && td < TREE_FAR) {                  // 3D trees, wherever the tree table has one within two cells
      vec2 t0 = floor(p.xz / 7.) + TREE_G0;
      bool tab = min(t0.x, t0.y) >= 1. && max(t0.x, t0.y) < float(TREE_N) - 1.;
      if (p.y - ty >= VEGMAX + 1.) { if (tab || d0 >= 0.03) s.vd = p.y - ty - VEGMAX; }   // above every tree: a bound only
      else {
        float mh = tab ? texelFetch(tTree, ivec2(t0), 0).b * 25.5 : (d0 >= 0.03 ? VEGMAX : 0.);
        if (mh > 0.5) s.vd = treeSD(p, s.gy, mh, tab, s.tc, s.tp);
        else if (tab) s.vd = 10.85;
      }
    } else if (s.dens >= 0.03)
      s.vy = p.y >= ty + VEGMAX + 1. ? ty + VEGMAX : canopyTop(p.xz, s.gy, s.dens, s.conF, s.tc, s.tp);
  }
  return s;
}
float topOf(TS s) { return max(max(s.gy, s.wy), s.vy); }

vec3 tetra(int i) { return 0.5773 * (2. * vec3(float(((i + 3) >> 1) & 1), float((i >> 1) & 1), float(i & 1)) - 1.); }

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
// otherwise inlines it once per use (march, refine, normals, shadow, reflection) and takes minutes.
#define ST_MARCH 0
#define ST_BISECT 1
#define ST_ONORM 2
#define ST_GNORM 3
#define ST_SHINIT 4
#define ST_SHADOW 5
#define ST_RINIT 6
#define ST_REFL 7
#define ST_RNORM 8
#define ST_DONE 9

// mat: 0 sky, 1 ground, 2 water, 3 vegetation, 4 castle, 5 building, 6 wall/hedge, 7 special
// reflection rmat: 0 sky, 1 ground, 3 canopy, 4 castle
struct Res {
  float t; int mat; float aux; float ctop; TS s; vec3 n; vec3 wn; float sh; float it;
  vec3 rd; int rmat; float rt; vec3 rp; vec3 rn; float rconF; float rctop;
};

Res trace(vec3 ro, vec3 rd) {
  Res r;
  r.t = TMAX; r.mat = 0; r.aux = 0.; r.ctop = 0.; r.n = vec3(0., 1., 0.); r.wn = r.n; r.sh = 1.; r.it = 0.;
  r.rd = rd; r.rmat = 0; r.rt = 0.; r.rp = vec3(0.); r.rn = r.n; r.rconF = 0.; r.rctop = 0.;
  int maxM = uQual == 0 ? 170 : (uQual == 1 ? 250 : 360);
  int maxS = uQual == 0 ? 36 : (uQual == 1 ? 60 : 96);
  int maxR = (uQual == 0 || uDetail == 0 || (uDbg & 2) != 0) ? 0 : (uQual == 1 ? 64 : 100);
  int st = ST_MARCH, cnt = 0;
  float t = 0.05, tPrev = 0.05, ta = 0., tb = 0., e = 0., g0 = 0., h1 = 0., stt = 0., rtt = 0.;
  vec3 hp = vec3(0.), nacc = vec3(0.), so = vec3(0.), rro = vec3(0.);
  if (ro.y > CEIL) { if (rd.y >= 0.) st = ST_DONE; else t = (ro.y - CEIL) / -rd.y; }
  int nIt = maxM + maxS + maxR + 16;
  for (int i = ZERO; i < nIt; i++) {
    // ---- transitions that need no scene sample
    if (st == ST_SHINIT) {
      vec2 dd = hp.xz - r.s.tc.xy;                   // tree hits: start outward from the crown, not straight up
      vec3 vo = r.s.tp.w > 0. ? normalize(vec3(dd.x, max(length(dd), 0.5) * 0.6, dd.y)) : vec3(0., 0.6, 0.);
      so = hp + (r.mat == 3 ? vo : (r.mat == 2 ? vec3(0., 0.1, 0.) : r.n * (0.04 + r.t * 0.0008))) + uSunDir * 0.05;
      stt = 0.12 + r.t * 0.0015; cnt = 0;
      r.sh = uSunDir.y < -0.03 ? 0. : 1.;
      st = (uSunDir.y > -0.03 && (uDbg & 1) == 0) ? ST_SHADOW : ST_RINIT;
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
    // ---- where to sample and what the sample needs (vm: vegetation mode, om: 0 none, 1 all objects, 2 castle only)
    vec3 p = vec3(0.); float td = 0.; int vm = 0, om = 0;
    if (st == ST_MARCH) { p = ro + rd * t; td = t; vm = uDetail == 1 ? 1 : 2; om = (uDbg & 4) == 0 ? 1 : 0; }
    else if (st == ST_BISECT) { t = 0.5 * (ta + tb); p = ro + rd * t; td = t; vm = uDetail == 0 ? 2 : (t < TREE_FAR ? 0 : 1); }
    else if (st == ST_ONORM) { p = hp + tetra(cnt) * e; td = r.t; om = 1; }
    else if (st == ST_GNORM || st == ST_RNORM) { p = hp + (cnt == 0 ? vec3(e, 0., 0.) : vec3(0., 0., e)); td = st == ST_GNORM ? r.t : 1e5; }
    else if (st == ST_SHADOW) { p = so + uSunDir * stt; td = r.t; vm = (r.t < 900. && uDetail == 1) ? 1 : 0; om = 1; }
    else { p = rro + r.rd * rtt; td = 1e5; vm = 2; om = 2; }
    // ---- the single scene evaluation
    TS s = evalTop(p, td, vm);
    float dO = 1e5, ao = 0., co = 0.; int mo = 0;
    if (om > 0) { if (s.g.ed > 0.) dO = objD(p, s.g, s.gy, om == 1, mo, ao, co); else dO = max(1. - s.g.ed, 1.); }
    float dy = p.y - topOf(s);
    // ---- advance
    if (st == ST_MARCH) {
      float eps = min(0.00045 * t + 0.01, 0.9);
      if (t > TMAX || (p.y > CEIL && rd.y > 0.)) st = ST_DONE;
      else if (dO < eps) {
        r.t = t; r.mat = mo; r.aux = ao; r.ctop = co; r.s = s; hp = p;
        e = mo == 4 ? max(0.3, t * 0.0008) : max(0.02, t * 0.0005); nacc = vec3(0.); cnt = 0; st = ST_ONORM;
      } else if (s.vd < eps) {                        // 3D tree: shading takes its normal from the tree's shape
        r.t = t; r.mat = 3; r.s = s; hp = p; cnt = 0; st = ST_SHINIT;
      } else if (dy < 0.) { ta = tPrev; tb = t; cnt = 0; st = ST_BISECT; }
      else if (++cnt >= maxM) {                      // out of steps: grazing ray, call it ground
        r.t = t; r.mat = 1; r.s = s; hp = p; g0 = s.gy; e = clamp(t * 0.0018, 0.3, 45.); cnt = 0; st = ST_GNORM;
      } else {
        tPrev = t;
        float k = mix(0.55, 0.85, smoothstep(900., 7000., t));
        t += min(min(max(k * dy, 0.0016 * t + 0.03), max(dO, eps)), max(0.8 * s.vd, max(eps, 0.0012 * t)));   // 0.8: trees ride the terrain slope
      }
    } else if (st == ST_BISECT) {
      if (dy < 0.) tb = t; else ta = t;
      if (++cnt >= 7) {
        float ty = max(s.gy, s.wy);
        r.t = t; r.s = s; hp = p; g0 = s.gy;
        r.mat = (s.vy > ty && s.tp.w != 0.) ? 3 : (s.wy >= s.gy ? 2 : 1);
        e = clamp(t * 0.0018, 0.3, 45.); cnt = 0; st = ST_GNORM;
      }
    } else if (st == ST_ONORM) {
      nacc += tetra(cnt) * dO;
      if (++cnt >= 4) { r.n = normalize(nacc); st = ST_SHINIT; }
    } else if (st == ST_GNORM || st == ST_RNORM) {
      if (cnt == 0) h1 = s.gy;
      else {
        vec3 n = normalize(vec3(g0 - h1, e, g0 - s.gy));
        if (st == ST_GNORM) { r.n = n; st = ST_SHINIT; } else { r.rn = n; st = ST_DONE; }
      }
      cnt++;
    } else if (st == ST_SHADOW) {
      float hh = min(min(dy, dO), s.tp.w > 0. ? s.vd : 1e5);   // tree bounds that aren't a crown's distance only limit the step
      r.sh = min(r.sh, 9. * hh / stt);
      stt += clamp(min(0.6 * min(dy, dO), 0.8 * s.vd), 0.12 + stt * 0.012, 350.);
      if (r.sh < 0.003 || p.y > CEIL || stt > 12000. || ++cnt >= maxS) { r.sh = smoothstep(0., 1., clamp(r.sh, 0., 1.)); st = ST_RINIT; }
    } else {                                         // ST_REFL
      float eps = min(0.03 + 0.002 * rtt, 0.9);
      if (dO < eps || dy < 0.) {
        r.rt = rtt; r.rp = p; r.rconF = s.conF; r.rctop = co;
        r.rmat = dO < eps ? 4 : (s.vy > max(s.gy, s.wy) ? 3 : 1);
        if (r.rmat == 4) st = ST_DONE;
        else { hp = p; g0 = s.gy; e = 8. + rtt * 0.004; cnt = 0; st = ST_RNORM; }
      } else {
        rtt += clamp(min(0.6 * dy, dO), 0.5 + rtt * 0.015, 300.);
        if (p.y > CEIL || rtt > 9000. || ++cnt >= maxR) st = ST_DONE;
      }
    }
  }
  return r;
}
