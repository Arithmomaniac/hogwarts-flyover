// ------------------------------------------------------------------ shading
vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0., 1.); }
//__LAMPS__

vec3 shade(vec3 ro, vec3 rd, Res r) {
  float t = r.t;
  vec3 p = ro + rd * t, L = uSunDir;
  bool primMiss = r.mat == 0, reflMiss = r.mat == 2 && r.rmat == 0;
  vec3 skyc = (primMiss || reflMiss) ? sky(primMiss ? ro : p, primMiss ? rd : r.rd) : vec3(0.);
  if (primMiss) return skyc;
  TS s = r.s;
  vec3 gn = r.n;
  float pix = t * 2. * uTanHalf / uRes.y;
  float sh = r.sh * cloudShadow(p);
  float depth = max(s.wy - s.gy, 0.);
  vec3 galb = vec3(0.);
  vec2 lb = vec2(0., 1.);                             // the local layer's shingle field and stream distance (locB)
  if (r.mat == 1 || (r.mat == 2 && depth < 4.)) {
    if (s.g.lw > 0.) lb = textureLod(tLocB, vec2(dot(uLocU, vec3(p.xz, 1.)), dot(uLocV, vec3(p.xz, 1.))), 0.).rg;
    galb = groundAlb(p, s, t, pix / max(abs(dot(gn, rd)), 0.25), gn, lb.x);
  }

  if (r.mat == 2) {                                   // loch / river: peaty water, Fresnel reflection, sun glint
    float fr = (0.02 + 0.98 * pow(1. - max(dot(-rd, r.wn), 0.), 5.)) * smoothstep(0., 0.6, depth);   // thin shore water: less mirror
    vec3 refl = skyc;
    if (r.rmat != 0) {
      vec3 q = r.rp, rc;
      if (r.rmat == 5) rc = r.rp;                     // a mesh, already lit by the mirrored mesh pass
      else {
        vec3 alb = r.rmat == 3 ? mix(vec3(0.04, 0.06, 0.02), vec3(0.022, 0.038, 0.026), r.rconF) : satAlb(q.xz, r.rt * 0.004);
        rc = alb * (uSunCol * max(dot(r.rn, L), 0.) + uSkyCol * (0.55 + 0.45 * r.rn.y));
      }
      refl = applyFog(rc, p, r.rd, r.rt);
    }
    vec3 body = mix(vec3(0.006, 0.0045, 0.0025), galb * vec3(0.55, 0.42, 0.25), exp(-depth * 0.8)) * (uSunCol * max(L.y, 0.) * sh + uSkyCol * 0.6);
    return mix(body, refl, fr) + uSunCol * sh * pow(max(dot(r.rd, L), 0.), 700.) * 40.;
  }

  vec3 n = gn, alb = vec3(0.1), emi = vec3(0.);
  float ao = 1., spec = 0., shin = 32., wrap = 0.;
  if (r.mat == 1) {
    alb = galb;
    if (s.g.lw > 0.) {
      float ss = lb.y * 25.5 - 12.8;
      float sm = (1. - smoothstep(-0.3, 0.5, ss)) * s.g.lw;
      alb = mix(alb, vec3(0.02, 0.017, 0.011), sm);
      spec = sm * 0.5; shin = 120.;
    }
    ao = mix(1., 0.55, s.dens);
    float rk = smoothstep(0.72, 0.42, gn.y) * (1. - smoothstep(0.3, 1.5, pix));
    if (rk > 0.) n = normalize(n + (vec3(vnoise(p.xz * 0.45 + p.y * 0.3), vnoise(p.zy * 0.45 + 5.), vnoise(p.xy * 0.45 + 9.)) - 0.5) * 0.9 * rk);
  } else if (r.mat == 3) {
    n = vegNormal(p, s, gn);
    float hue = s.tp.z;
    vec3 dec = mix(vec3(0.034, 0.058, 0.016), vec3(0.06, 0.074, 0.02), hue);
    dec = mix(dec, vec3(0.1, 0.075, 0.02), step(0.94, hue) * 0.7);   // a few trees already turning
    vec3 con = vec3(0.021, 0.037, 0.025) * (0.85 + 0.3 * hue);
    alb = s.tp.w < 0. ? mix(vec3(0.045, 0.064, 0.018), con, s.conF) : (s.tp.w > 1.5 ? vec3(0.05, 0.04, 0.03) : (s.tp.x > 0.5 ? con : dec));
    ao = s.tp.w < 0. ? 0.7 : 0.3 + 0.7 * clamp((p.y - s.gy) / max(s.tc.w, 1.), 0., 1.);
    wrap = 0.3;
  } else if (r.mat == 4) {                            // castle (meshes)
    bool mt = r.aux > 149.5;                          // mesh tower: re-ranked top in ctop, battlements unless the dome
    bool part = r.aux > 39.5 && r.aux < 49.5;         // mesh openings: 40 pane, 41 lit and 45 dimly lit at dusk, 42 oak, 43 their masonry lining, 44 steps
    bool cut = mt || part || (r.aux > 19.5 && r.aux < 39.5);   // real openings (mesh class 20 + roof type): no painted windows
    float top = r.ctop, rt = mt || part ? -1. : (r.aux > 19.5 ? r.aux - 20. : r.aux);
    int ti = mt ? int(r.aux - 150. + 0.5) : -1;
    vec2 bw = vec2(faceCoord(p, n, true) / 1.3, p.y / 0.62);
    bw.x += 0.5 * mod(floor(bw.y), 2.);
    alb = vec3(0.26, 0.228, 0.19) * (0.86 + 0.28 * hash12(floor(bw)) * (1. - smoothstep(0.15, 0.6, pix)));   // warm brown-grey
    alb *= 0.88 + 0.24 * vnoise(p.xz * 0.05 + p.y * 0.04);
    bool roof = false; float wx = faceCoord(p, n, true), wtop = top - 1.5;
    if (ti >= 0) {
      vec4 tw = uTw[ti]; float rf = uTx[ti].x, tt = mt ? r.ctop : tw.w;
      if (mt && (rf < 3.5 || rf > 4.5)) rf = 0.;
      vec2 q = p.xz - tw.xy;
      wx = atan(q.y, q.x) * tw.z; wtop = tt - 1.5;
      if (rf > 3.5 && rf < 4.5 && p.y > tt + 0.6) { alb = vec3(0.05, 0.085, 0.075); spec = 0.25; shin = 40.; roof = true; }
      else if (p.y > tt + 0.3 && ((rf > 0.5 && rf < 1.5) || rf > 4.5)) roof = true;
    } else if ((rt == 1. || rt == 3.) && n.y > 0.3) roof = true;
    else if (rt == 0. && n.y > 0.7) alb *= 0.7;
    if (roof && spec == 0.) { alb = vec3(0.045, 0.05, 0.06) * (0.85 + 0.3 * vnoise(p.xz * 0.6)); spec = 0.12; shin = 24.; }
    if (!roof && abs(n.y) < 0.3 && !cut) {
      float lit; float wm = windowMask(p, wx, LAKE + 39., wtop, vec2(3.2, 4.1), pix, lit);
      alb = mix(alb, vec3(0.012), wm * 0.85);
      spec = max(spec, wm * 0.6); shin = max(shin, wm * 90.);
      emi += wm * lit * uDusk * CANDLE * 0.2;
    }
    ao = 0.65 + 0.35 * smoothstep(LAKE + 35., LAKE + 47., p.y);
    if (part) {                                       // 40-43 and 45 sit recessed in the wall, where less of the sky reaches them
      bool pane = r.aux < 41.5 || (r.aux > 44.5 && r.aux < 45.5);   // 40 dark, 41 lit and 45 dimly lit at dusk
      if (r.aux < 43.5 || pane) ao *= 0.6;
      if (pane) {                                     // glass: dark, a mirror-like glint of the sun and a Fresnel sky
        alb = vec3(0.012); spec = 0.6; shin = 400.;
        float fres = 0.04 + 0.96 * pow(1. - max(dot(n, -rd), 0.), 5.);
        emi += fres * mix(uHorCol, uSkyCol, clamp(reflect(rd, n).y * 2. + 0.5, 0., 1.)) * ao + lampGlow(4, r.aux) * uDusk;
      }
      else if (r.aux < 42.5) alb = vec3(0.09, 0.06, 0.035);
    }
  } else if (r.mat == 5) {                            // Hogsmeade, Hagrid's hut, glasshouses, stands
    float kind = r.aux;
    bool roof = n.y > 0.25;
    vec3 wallC = vec3(0.3, 0.285, 0.26), roofC = vec3(0.05, 0.052, 0.06);
    if (kind == 0.) roofC = vec3(0.15, 0.115, 0.06) * (0.8 + 0.4 * vnoise(p.xz * 3.));
    else if (kind == 2.) { wallC = roofC = vec3(0.16, 0.2, 0.19); spec = 0.6; shin = 200.; }
    else if (kind == 3.) { wallC = vec3(0.19, 0.165, 0.13); roofC = vec3(0.09, 0.07, 0.045); }
    else if (kind == 4.) { wallC = vec3(0.085, 0.07, 0.055); roofC = vec3(0.055, 0.047, 0.042); }
    else if (kind == 5.) { wallC = standCol(p.xz); roofC = vec3(0.16, 0.12, 0.08); }
    else if (kind > 5.5) wallC = vec3(0.24, 0.225, 0.2);
    alb = roof ? roofC : wallC;
    if (!roof && kind != 2. && kind != 5.) {
      float lit; float wm = windowMask(p, faceCoord(p, n, true), s.gy + 0.9, s.gy + 7., vec2(2.7, 2.9), pix, lit);
      alb = mix(alb, vec3(0.015), wm * 0.8);
      emi += wm * lit * uDusk * DIMLAMP * 0.2;
    }
  } else if (r.mat == 6) {                            // boundary wall / hedges
    if (r.aux > 0.5) alb = vec3(0.13, 0.125, 0.115) * (0.8 + 0.4 * vnoise(p.xz * 2. + p.y));
    else { alb = vec3(0.03, 0.058, 0.02); n = normalize(n + (vec3(vnoise(p.xz * 3.), vnoise(p.zy * 3.), vnoise(p.xy * 3.)) - 0.5) * 0.8); wrap = 0.3; }
  } else {                                            // set pieces
    float sid = r.aux;
    if (sid < 1.5) { alb = vec3(0.55, 0.38, 0.09); spec = 0.8; shin = 60.; }
    else if (sid < 4.5) alb = vec3(0.045, 0.035, 0.025);
    else if (sid < 5.5) alb = vec3(0.03, 0.025, 0.02);
    else if (sid < 6.5) alb = vec3(0.21, 0.2, 0.185);
    else if (sid < 7.5) { alb = vec3(0.62, 0.61, 0.58); spec = 0.3; shin = 80.; }
    else {                                            // the families' set pieces (src/materials)
      int k = int(sid + 0.5);
      //__SET_PIECES_P3__
    }
  }

  float ndl = dot(n, L);
  float dif = max((ndl + wrap) / (1. + wrap), 0.);
  vec3 amb = uSkyCol * (0.5 + 0.5 * n.y) * 0.95 + uHorCol * 0.12 * (1. - n.y)
           + uSunCol * max(L.y, 0.) * 0.06 * (1. - n.y) * vec3(0.8, 1., 0.6);       // bounce from sunlit ground
  vec3 col = alb * (uSunCol * dif * sh + amb * ao);
  if (spec > 0.) {
    vec3 hv = normalize(L - rd);
    col += uSunCol * sh * spec * pow(max(dot(n, hv), 0.), shin) * step(0., ndl) * (shin + 8.) / 25.;
  }
  return col + emi;
}

#if defined(SPLIT_P3) || defined(SUN_PASS)
// sun visibility of p against the mesh casters: the finest layer whose footprint holds p, four bilinear PCF taps.
// Layers hold every caster over their footprint, so meshes outside the view still shade what the view sees. The
// shading pass looks them up for each pixel, the sun cache for each 7 m cell.
float meshSun(vec3 p) {
  for (int k = uShN - 1; k >= 0; k--) {
    vec4 c = uShM[k] * vec4(p, 1.);
    if (max(abs(c.x), abs(c.y)) < 0.995) {
      vec3 s = c.xyz * 0.5 + 0.5;
      float z = min(s.z, 1.) - uShP[k].z, d = 0.5 * uShP[k].x;
      return 0.25 * (texture(tMeshSh, vec4(s.xy + vec2(-d, -d), float(k), z)) + texture(tMeshSh, vec4(s.xy + vec2(d, -d), float(k), z))
                   + texture(tMeshSh, vec4(s.xy + vec2(-d, d), float(k), z)) + texture(tMeshSh, vec4(s.xy + vec2(d, d), float(k), z)));
    }
  }
  return 1.;
}
#endif

#ifdef TREE_PASS
void main() {                                         // one texel per tree cell: the table treeSD reads
  vec2 cc = floor(gl_FragCoord.xy) - TREE_G0;
  float d = 0., cf = 0., mh = 0., r;
  for (int k = ZERO; k < 25; k++) {                  // own cell (k == 12) and the tallest tree within two cells
    vec2 c = cc + vec2(float(k % 5), float(k / 5)) - 2.;
    vec4 h = hash42(c);
    float conF, dn = floor(treeDens((c + 0.25 + 0.5 * h.yz) * 7., conF) * 255. + 0.5) / 255.;   // as stored
    conF = floor(conF * 255. + 0.5) / 255.;
    if (k == 12) { d = dn; cf = conF; }
    if (h.x < dn) mh = max(mh, treeH(h, dn, h.w < conF, r));
  }
  outColor = vec4(d, cf, (mh + 0.1) / 25.5, 1.);
}
#elif defined(CONE_PASS)
// one texel per 8x8 pixel block: how far a cone round the block's centre ray, radius r(t) = 8 pixel footprints, provably
// crosses free air. A segment [t, t1] is accepted only if its whole capsule (dilated by r(t1)) stays clear of the local
// grid's edge and lies above the static max-height grid (terrain, water, detail relief, tallest vegetation) of every cell
// its footprint touches. Only accepted distances are written. Meshes are not in it: P1 ends its march at their depth.
float cgCell(vec4 G, ivec2 c) {                       // G: first column, origin, cell size, cells per side
  uvec4 b = uvec4(texelFetch(tTree, ivec2(int(G.x) + c.x, TREE_N + c.y), 0) * 255. + 0.5);
  return uintBitsToFloat(b.r | (b.g << 8) | (b.b << 16) | (b.a << 24));
}
float cgMax(vec4 G, vec2 lo, vec2 hi) {
  ivec2 n = ivec2(int(G.w) - 1);
  ivec2 a = clamp(ivec2(floor((lo - G.y) / G.z)), ivec2(0), n), b = clamp(ivec2(floor((hi - G.y) / G.z)), ivec2(0), n);
  if (any(greaterThan(b - a, ivec2(3)))) return 1e30;
  float m = -1e30;
  for (int j = ZERO; j < 4; j++) for (int i = ZERO; i < 4; i++) {
    ivec2 c = a + ivec2(i, j);
    if (c.x <= b.x && c.y <= b.y) m = max(m, cgCell(G, c));
  }
  return m;
}
// Phase 0 finds the verified free distance tg. Phase 1 then replays the main pass's exact ST_MARCH step rule along the
// same centre ray and keeps the last baseline sample t_k <= tg - r(tg) with its step index k, so the main pass resumes
// the baseline's march state (position and maxM budget) instead of restarting the count. A replay that exhausts the
// budget before tg writes 0 (no skip); k lives in the low 9 mantissa bits, whose truncation only lowers t_k.
void main() {
  vec2 b = floor(gl_FragCoord.xy);
  vec2 uv = (2. * (b * 8. + 4. + uJitter) - uRes) / uRes.y;
  vec3 rd = normalize(uCamRot * vec3(uv * uTanHalf, 1.)), ro = uCamPos;
  float rk = 16. * uTanHalf / uRes.y, ax = max(abs(rd.x), abs(rd.z));
  int maxM = uQual == 0 ? 170 : (uQual == 1 ? 250 : 360);
  int vmM = uDetail == 1 ? 1 : 2;
  float t0 = 0.05;
  int ph = 0, gi = 0, cnt = 0, k = -1, kp = -1;      // ph 0 grid march, 1 baseline replay, 2 done
  if (ro.y > CEIL) { if (rd.y >= 0.) ph = 2; else t0 = (ro.y - CEIL) / -rd.y; }
  float t = t0, L = 2., Lo = 0., tr = t0, tlim = 0., tk = 0., tkp = 0.;
  bool fresh = true;
  for (int i = ZERO; i < 530; i++) {                 // 160 grid steps + at most 360 replay samples
    if (ph == 0 && (t >= TMAX || gi >= 160)) { float tg = min(t, TMAX); tlim = tg - rk * tg; ph = 1; }
    if (ph == 1 && tr > tlim) break;
    if (ph == 2) break;
    bool rep = ph == 1;
    float ts = rep ? tr : t;
    vec3 p = ro + rd * ts;
    float dE = 1e5, dy = 1e5, vd = 1e5;
    if (rep || fresh) {                               // the single scene evaluation of this pass
      TS s = evalTop(p, ts, rep ? vmM : 0);
      dE = s.g.ed > 0. ? 1e5 : max(1. - s.g.ed, 1.);  // the local grid's edge, as trace() bounds its step by it
      dy = p.y - topOf(s); vd = s.vd;
    }
    if (rep) {                                        // exactly trace()'s ST_MARCH advance
      tkp = tk; kp = k; tk = tr; k = cnt;
      float eps = min(0.00045 * tr + 0.01, 0.9);
      if (tr > TMAX || (p.y > CEIL && rd.y > 0.)) break;
      if (vd < eps || dy < 0.) { tk = tkp; k = kp; break; }   // keep the bisection bracket's start sample
      if (++cnt >= maxM) { k = -1; break; }          // baseline exhausts before tg: no skip for this block
      float km = mix(0.55, 0.85, smoothstep(900., 7000., tr));
      tr += min(min(max(km * dy, 0.0016 * tr + 0.03), max(dE, eps)), max(0.8 * vd, max(eps, 0.0012 * tr)));
      continue;
    }
    gi++;
    if (fresh) {                                      // clearance round the accepted point: the local grid's edge
      Lo = (0.75 * dE - rk * t - 1.) / (1. + rk);    // capsule within 0.75 dE - 1 m of p
      fresh = false;
    }
    L = min(L, min(Lo, (3. * CG_FAR.z - 2. * rk * t) / (ax + 2. * rk)));   // footprint spans <= 4 far cells
    if (L < max(0.5, 0.004 * t)) { gi = 160; continue; }
    float t1 = t + L, r1 = rk * t1;
    vec3 p1 = ro + rd * t1;
    vec2 lo = min(p.xz, p1.xz) - r1, hi = max(p.xz, p1.xz) + r1;
    float ext = max(hi.x - lo.x, hi.y - lo.y);
    bool inF = max(max(-lo.x, -lo.y), max(hi.x, hi.y)) < -CG_FINE.y;
    vec4 G = (ext <= 3. * CG_FINE.z && inF) ? CG_FINE : (ext <= 3. * CG_MID.z ? CG_MID : CG_FAR);
    if (min(p.y, p1.y) - r1 - 1. > cgMax(G, lo, hi)) { t = t1; L *= 2.; fresh = true; }
    else L *= 0.5;
  }
  uint u = (k >= 0 && k < 512 && tk <= TMAX) ? (floatBitsToUint(tk) & ~511u) | uint(k) : 0u;
  outColor = vec4(uvec4(u, u >> 8, u >> 16, u >> 24) & 255u) / 255.;
}
#elif defined(SUN_PASS)
void main() {                                         // alpha only: sun visibility of the far surface at each tree-cell centre
  vec2 xz = (floor(gl_FragCoord.xy) - TREE_G0 + 0.5) * 7.;
  TS s = evalTop(vec3(xz.x, -1e4, xz.y), SUN_NEAR, 2);   // as the primary march sees it beyond SUN_NEAR (no micro-relief)
  float ty = max(s.gy, s.wy);
  int mat = (s.vy > ty && s.tp.w != 0.) ? 3 : (s.wy >= s.gy ? 2 : 1);
  vec3 hp = vec3(xz.x, topOf(s), xz.y);
  float e = clamp(SUN_TREF * 0.0018, 0.3, 45.);
  vec3 n = normalize(vec3(s.gy - sampleGround(xz + vec2(e, 0.)).bed, e, s.gy - sampleGround(xz + vec2(0., e)).bed));
  vec3 so = hp + (mat == 3 ? vec3(0., 0.6, 0.) : (mat == 2 ? vec3(0., 0.1, 0.) : n * (0.04 + SUN_TREF * 0.0008))) + uSunDir * 0.05;
  float stt = 0.12 + SUN_TREF * 0.0015, sh = 1.;
  int maxS = uQual == 0 ? 36 : (uQual == 1 ? 60 : 96);
  for (int i = ZERO; i < 96; i++) {
    vec3 p = so + uSunDir * stt;
    TS q = evalTop(p, SUN_TREF, 0);
    float dE = q.g.ed > 0. ? 1e5 : max(1. - q.g.ed, 1.);   // the local grid's edge only limits the step
    float dy = p.y - topOf(q);
    sh = min(sh, 9. * min(dy, q.tp.w > 0. ? q.vd : 1e5) / stt);   // tree bounds that aren't a crown's distance only limit the step
    stt += clamp(min(0.6 * min(dy, dE), 0.8 * q.vd), 0.12 + stt * 0.012, 350.);
    if (sh < 0.003 || p.y > CEIL || stt > 12000. || i + 1 >= maxS) break;
  }
  outColor = vec4(0., 0., 0., min(smoothstep(0., 1., clamp(sh, 0., 1.)), meshSun(so)));
}
#elif defined(SPLIT_P1)
void main() {
  vec2 uv = (2. * (gl_FragCoord.xy + uJitter) - uRes) / uRes.y;
  vec3 rd = normalize(uCamRot * vec3(uv * uTanHalf, 1.));
  vec3 ro = uCamPos;
  Res r = trace(ro, rd);
  // G-buffer (float32): code packs mat, rmat, shadow flag and loop iterations; material-specific slots
  float code = float(r.mat + 8 * r.rmat + (r.shf ? 64 : 0)) + 128. * r.it;
  gOut0 = vec4(r.t, code, r.mat == 3 ? r.s.conF : (r.mat == 2 ? r.rconF : 0.), r.mat == 2 ? r.rt : 0.);
  gOut1 = vec4(r.n, r.s.gy);
  gOut2 = r.mat == 3 ? r.s.tc : vec4(r.s.g.lw, r.s.g.la.r, r.s.dens, r.s.wy);
  gOut3 = r.mat == 3 ? r.s.tp : (r.mat == 2 ? vec4(r.wn, r.rp.x) : vec4(r.s.g.luv, 0., 0.));
  gOut4 = vec4(r.rd, r.rp.z);
  gOut5 = vec4(r.rn, r.rp.y);
  gOut6 = r.shf ? vec4(r.so, r.stt) : vec4(r.sh, 0., 0., -1.);   // no march: the sun term trace() settled on
}
#elif defined(SPLIT_SH)
void main() {                                         // the soft-shadow march from where trace() left the hit (the mesh casters' shadows are the shading pass's)
  ivec2 ip = min(ivec2(gl_FragCoord.xy) * SDIV + SDIV / 2, ivec2(uRes) - 1);   // SDIV > 1: the block's centre pixel
  vec4 a6 = texelFetch(tG6, ip, 0);
  if (a6.w < 0.) { outColor = vec4(a6.x, 0., 0., 0.); return; }
  float rt = texelFetch(tG0, ip, 0).x;
  vec3 so = a6.xyz;
  float stt = a6.w, sh = 1., it = 0.;                // only marched when uSunDir.y > -0.03, where trace starts at 1
  int maxS = uQual == 0 ? 36 : (uQual == 1 ? 60 : 96);
  int cnt = 0;
  for (int i = ZERO; i < maxS; i++) {
    it += 1.;
    vec3 p = so + uSunDir * stt; float td = rt; int vm = (rt < 900. && uDetail == 1) ? 1 : 0;
    TS s = evalTop(p, td, vm);
    float dE = s.g.ed > 0. ? 1e5 : max(1. - s.g.ed, 1.);   // the local grid's edge only limits the step
    float dy = p.y - topOf(s);
    sh = min(sh, 9. * min(dy, s.tp.w > 0. ? s.vd : 1e5) / stt);   // tree bounds that aren't a crown's distance only limit the step
    stt += clamp(min(0.6 * min(dy, dE), 0.8 * s.vd), 0.12 + stt * 0.012, 350.);
    if (sh < 0.003 || p.y > CEIL || stt > 12000. || ++cnt >= maxS) { sh = smoothstep(0., 1., clamp(sh, 0., 1.)); break; }
  }
  outColor = vec4(sh, it, 0., 0.);
}
#elif defined(SPLIT_P3)
void main() {
  vec2 uv = (2. * (gl_FragCoord.xy + uJitter) - uRes) / uRes.y;
  vec3 rd = normalize(uCamRot * vec3(uv * uTanHalf, 1.));
  vec3 ro = uCamPos;
  ivec2 ip = ivec2(gl_FragCoord.xy);
  vec4 a0 = texelFetch(tG0, ip, 0), a1 = texelFetch(tG1, ip, 0), a2 = texelFetch(tG2, ip, 0);
  vec4 a3 = texelFetch(tG3, ip, 0), a4 = texelFetch(tG4, ip, 0), a5 = texelFetch(tG5, ip, 0);
  vec4 o6 = texelFetch(tG6, ip, 0);
#if SDIV == 1
  vec2 shv = texelFetch(tSh, ip, 0).xy;
#else
  // joint-bilateral upsample: the 2x2 nearest block centres, bilinear-weighted, kept when they were marched and agree
  // in depth (2% + 5 cm), normal (cos > 0.8) and material; else the block centre nearest in depth (its marched or settled
  // sun term). Pixels whose own sun term was settled without a march keep it exactly.
  vec2 shv = vec2(1., 0.);
  if (o6.w < 0.) shv.x = o6.x;
  else {
    ivec2 lr = textureSize(tSh, 0) - 1, fr = ivec2(uRes) - 1;
    vec2 f = (vec2(ip) + 0.5) / float(SDIV) - 0.5, w = fract(f);
    ivec2 q0 = ivec2(floor(f));
    float tol = 0.02 * a0.x + 0.05, acc = 0., ws = 0., bd = 1e30;
    int om = int(a0.y) & 7;
    for (int j = 0; j < 4; j++) {
      ivec2 o = ivec2(j & 1, j >> 1), q = clamp(q0 + o, ivec2(0), lr), rp = min(q * SDIV + SDIV / 2, fr);
      vec4 b0 = texelFetch(tG0, rp, 0), b6 = texelFetch(tG6, rp, 0);
      float sq = b6.w < 0. ? b6.x : texelFetch(tSh, q, 0).x, dt = abs(b0.x - a0.x);
      if (dt < bd) { bd = dt; shv.x = sq; }
      if (b6.w >= 0. && dt < tol && (int(b0.y) & 7) == om && dot(texelFetch(tG1, rp, 0).xyz, a1.xyz) > 0.8) {
        float bw = max((o.x == 1 ? w.x : 1. - w.x) * (o.y == 1 ? w.y : 1. - w.y), 1e-4);
        acc += bw * sq; ws += bw;
      }
    }
    if (ws > 0.) shv.x = acc / ws;
  }
#endif
  // the mesh casters' shadow for this pixel's own shadow start: a thin caster's shadow narrower than a shadow-pass block
  // would otherwise be seen only where it crosses a block's centre, and break into dots
  if (o6.w >= 0.) shv.x = min(shv.x, meshSun(o6.xyz));
  int ic = int(a0.y);
  Res r;
  r.t = a0.x; r.mat = ic & 7; r.rmat = (ic >> 3) & 7; r.it = float(ic >> 7) + shv.y;
  r.aux = a0.z; r.ctop = a0.w; r.rconF = a0.z; r.rt = a0.w;
  r.n = a1.xyz; r.sh = shv.x;
  r.s.g.bed = 0.; r.s.g.wd = 0.; r.s.g.ed = 0.; r.s.g.la = vec4(0.); r.s.g.lw = 0.; r.s.g.luv = vec2(0.);
  r.s.gy = a1.w; r.s.wy = 0.; r.s.vy = 0.; r.s.vd = 0.; r.s.dens = 0.; r.s.conF = 0.; r.s.tc = vec4(0.); r.s.tp = vec4(0.);
  r.wn = vec3(0.); r.rd = a4.xyz; r.rp = vec3(a3.w, a5.w, a4.w); r.rn = a5.xyz;
  if (r.mat == 3) { r.s.tc = a2; r.s.tp = a3; r.s.conF = a0.z; }
  else {
    r.s.g.lw = a2.x; r.s.g.la.r = a2.y; r.s.dens = a2.z; r.s.wy = a2.w;
    if (r.mat == 2) r.wn = a3.xyz; else r.s.g.luv = a3.xy;
  }
  if (uMeshRefl != 0 && r.mat == 2 && abs(r.s.wy - LAKE) < 0.25) {
    // the mirrored pass holds the flat-mirror image (the scene mirrored about the lake, seen from this camera); the
    // rippled ray leaves the water point, not the camera, so look up where the camera sees the rippled ray's point at
    // the distance the flat image gives (its own direction when the flat image is empty), and keep the mesh where it
    // is nearer than the reflected terrain or sky
    ivec2 rs = textureSize(tMeshRefl, 0);
    vec4 f = texelFetch(tMeshRefl, min(ivec2(gl_FragCoord.xy * vec2(rs) / uRes), rs - 1), 0);
    vec3 mr = vec3(r.rd.x, -r.rd.y, r.rd.z);
    vec3 c = transpose(uCamRot) * (f.a > r.t ? rd * r.t + mr * (f.a - r.t) : mr);
    if (c.z > 1e-4) {
      vec2 sp = (c.xy / (c.z * uTanHalf) * uRes.y + uRes) * 0.5 - uJitter;
      ivec2 rq = clamp(ivec2(sp * vec2(rs) / uRes), ivec2(0), rs - 1);
      vec4 m = texelFetch(tMeshRefl, rq, 0);
      float rt = m.a - r.t;
      if (m.a > 0. && rt > 0. && (r.rmat == 0 || rt < r.rt)) { r.rmat = 5; r.rt = rt; r.rp = m.rgb; }
    }
  }
  if ((uDbg & 128) != 0) { outColor = vec4(r.rmat == 5 ? 1. : 0., r.rt, float(r.rmat), -1.); return; }
  if ((uDbg & 16) != 0) { outColor = vec4(vec3(r.it / 400.), -1.); return; }
  if ((uDbg & 32) != 0) { outColor = vec4(vec3(r.t / 8000.), -1.); return; }
  if ((uDbg & 64) != 0) { outColor = vec4(vec3(r.sh), -1.); return; }   // the upsampled sun term
  vec3 col = shade(ro, rd, r);
  if (r.mat != 0) col = applyFog(col, ro, rd, r.t);
  col = pow(aces(col * uExposure), vec3(1. / 2.2));
  outColor = vec4(col, r.mat == 0 ? -1. : r.t);
}
#endif
