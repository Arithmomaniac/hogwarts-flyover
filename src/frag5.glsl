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
  float ao = 1., spec = 0., shin = 32., wrap = 0., burn = 0.;
  if (r.mat == 1) {
    alb = galb;
    if (s.g.lw > 0.) {
      float ss = lb.y * 25.5 - 12.8;
      float sm = (1. - smoothstep(-0.3, 0.5, ss)) * s.g.lw;
      alb = mix(alb, vec3(0.02, 0.017, 0.011), sm);
      spec = sm * 0.5; shin = 120.;
      burn = sm;
    }
    ao = mix(1., 0.55, s.dens);
    n = rockBump(p, n, pix);
  } else if (r.mat == 3) {
    n = vegNormal(p, s, gn);
    float hue = s.tp.z, crown = 1.;
    if (s.tp.w < 0. && (uDbg & 524288) == 0) {
      // the far canopy sheet, where the trees are too small to draw: each 7 m cell of the tree table takes a crown's
      // hue and light, as the 3D trees do, while a cell spans a pixel or two (nearer, as lite detail draws the sheet,
      // the cells would be a quilt), fading to their mean as it shrinks under one; clumps of a few dozen metres shade
      // it beyond that, so the Forest still reads as tree cover rather than a lawn (debug bit 524288: a smooth sheet,
      // and the burns as painted beds)
      vec2 cell = floor(p.xz / 7.);
      float k = smoothstep(1.5, 3.5, pix) * (1. - smoothstep(7., 21., pix));
      hue = mix(0.5, hash12(cell + 17.3), k);
      crown = mix(1., 0.5 + hash12(cell + 41.9), k) * (0.78 + 0.36 * vnoise(p.xz * 0.025));   // the 3D trees' mean light
    }
    vec3 dec = mix(vec3(0.034, 0.058, 0.016), vec3(0.06, 0.074, 0.02), hue);   // every broadleaf in leaf (trees-in-leaf)
    vec3 con = vec3(0.021, 0.037, 0.025) * (0.85 + 0.3 * hue);
    alb = s.tp.w < 0. ? mix(dec, con, s.conF) * crown : (s.tp.w > 1.5 ? vec3(0.05, 0.04, 0.03) : (s.tp.x > 0.5 ? con : dec));
    ao = s.tp.w < 0. ? 0.7 : 0.3 + 0.7 * clamp((p.y - s.gy) / max(s.tc.w, 1.), 0., 1.);
    wrap = 0.3;
#ifdef MESH_TREES
    if (s.tp.w > 2.5) {                               // a mesh tree (tp.y its species): the kit's bark, turning to its upper
      int sp = int(s.tp.y + 0.5);                     // colour by its aux (conF) and to the drawn crowns' far off; its leaves
      float hf = clamp((p.y - s.gy) / max(s.tc.w, 1.), 0., 1.);   // the drawn crown's colour (its hue and all),
      vec3 bark = mix(uTreeKit[3 * sp], uTreeKit[3 * sp + 1], s.conF);   // darkening into the crown by its aux
      alb = s.tp.w > 3.5 ? mix(bark, vec3(0.05, 0.04, 0.03), smoothstep(uTreeFar.x, uTreeFar.y, t)) : (s.tp.x > 0.5 ? con : dec);
      ao = s.tp.w > 3.5 ? 0.5 + 0.5 * hf : (0.45 + 0.55 * hf) * (1. - 0.5 * s.conF);
    }
#endif
  } else if (r.mat == 4) {                            // castle (meshes)
    bool mt = r.aux > 149.5;                          // mesh tower: re-ranked top in ctop, battlements unless the dome
    bool part = r.aux > 39.5 && r.aux < 49.5;         // mesh openings: 40 pane, 41 lit and 45 dimly lit at dusk, 42 oak, 43 their masonry lining, 44 steps, 47 iron
    bool cut = mt || part || (r.aux > 19.5 && r.aux < 39.5);   // real openings (mesh class 20 + roof type): no painted windows
    float top = r.ctop, rt = mt || part ? -1. : (r.aux > 19.5 ? r.aux - 20. : r.aux);
    int ti = mt ? int(r.aux - 150. + 0.5) : -1;
    vec2 bw = vec2(faceCoord(p, n, true) / 1.3, p.y / 0.62);
    bw.x += 0.5 * mod(floor(bw.y), 2.);
    alb = vec3(0.26, 0.228, 0.19) * (0.86 + 0.28 * hash12(floor(bw)) * (1. - smoothstep(0.15, 0.6, pix)));   // warm brown-grey
    alb *= 0.88 + 0.24 * vnoise(p.xz * 0.05 + p.y * 0.04);
    alb *= (0.93 + 0.14 * vnoise(vec2((bw.x - 0.5 * mod(floor(bw.y), 2.)) * 0.8, p.y * 0.035)))   // rain-streaked down the walls (unbonded)
         * mix(0.8, 1., smoothstep(0., 2.5, p.y - s.gy));               // and soiled at their feet
    bool roof = false; float wx = faceCoord(p, n, true), wtop = top - 1.5, sky = 1.;
    if (ti >= 0) {
      vec4 tw = uTw[ti]; float rf = uTx[ti].x, tt = mt ? r.ctop : tw.w;
      if (mt && (rf < 3.5 || rf > 4.5)) rf = 0.;
      vec2 q = p.xz - tw.xy;
      wx = atan(q.y, q.x) * tw.z; wtop = tt - 1.5;
      if (rf > 3.5 && rf < 4.5 && p.y > tt + 0.6) { alb = vec3(0.05, 0.085, 0.075); spec = 0.25; shin = 40.; roof = true; }
      else if (p.y > tt + 0.3 && ((rf > 0.5 && rf < 1.5) || rf > 4.5)) roof = true;
      else if (n.y > 0.7) sky = 0.6;                 // a tower's roof walk, inside its battlements
    } else if ((rt == 1. || rt == 3.) && n.y > 0.3) roof = true;
    else if (rt == 0. && n.y > 0.7) { alb *= 0.7; sky = 0.6; }   // a wall-walk, its parapets shading it from much of the sky
    if (roof && spec == 0.) {                         // slate: courses along the eaves, each slate a shade of its own
      vec2 sl = vec2(dot(p.xz, normalize(vec2(-n.z, n.x) + 1e-6)) / 0.3, p.y / 0.16);
      sl.x += 0.5 * mod(floor(sl.y), 2.);
      float near = 1. - smoothstep(0.08, 0.35, pix);
      alb = vec3(0.072, 0.074, 0.082) * (0.85 + 0.3 * vnoise(p.xz * 0.6)) * (1. + near * (0.3 * hash12(floor(sl)) - 0.15))
          * (1. - near * 0.35 * (1. - smoothstep(0., 0.18, fract(sl.y))));   // each course's lower edge in its own shadow
      spec = 0.12; shin = 24.;
    }
    if (!roof && abs(n.y) < 0.3 && !cut) {
      float lit; float wm = windowMask(p, wx, LAKE + 39., wtop, vec2(3.2, 4.1), pix, lit);
      alb = mix(alb, vec3(0.012), wm * 0.85);
      spec = max(spec, wm * 0.6); shin = max(shin, wm * 90.);
      emi += wm * lit * uDusk * CANDLE * 0.2;
    }
    ao = (0.65 + 0.35 * smoothstep(LAKE + 35., LAKE + 47., p.y)) * sky;
    ao *= 1. - 0.45 * clamp((-n.y - 0.2) * 2.5, 0., 1.);   // turned down (a corbel, a soffit): it sees the far ground, not the sky
    if (part) {                                       // 40-43 and 45 sit recessed in the wall, where less of the sky reaches them
      bool pane = r.aux < 41.5 || (r.aux > 44.5 && r.aux < 45.5);   // 40 dark, 41 lit and 45 dimly lit at dusk
      if (r.aux < 43.5 || pane) ao *= 0.6;
      if (r.aux > 42.5 && r.aux < 43.5 && n.y > 0.5) {   // a sill deep in its reveal, a floor under its vault: lit from the
        ao *= 0.4; n = normalize(vec3(-rd.x, 0.3, -rd.z));   // opening rather than from the sky overhead
      }
      if (pane) {                                     // glass: dark, a mirror-like glint of the sun and a Fresnel sky
        alb = vec3(0.012); spec = 0.6; shin = 400.;
        float fres = 0.04 + 0.96 * pow(1. - max(dot(n, -rd), 0.), 5.);
        emi += fres * mix(uHorCol, uSkyCol, clamp(reflect(rd, n).y * 2. + 0.5, 0., 1.)) * ao + lampGlow(4, r.aux) * uDusk;
      }
      else if (r.aux < 42.5) alb = vec3(0.09, 0.06, 0.035);
      else if (r.aux > 46.5 && r.aux < 47.5) { alb = vec3(0.025, 0.023, 0.022); spec = 0.35; shin = 60.; }   // iron
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
  if (burn > 0. && (uDbg & 524288) == 0) {
    // a burn's running water over its peaty bed: the sky it mirrors through its riffles, as the lake's Fresnel, and the
    // sun's glints off them; the riffles (still, a metre or so across) flatten where a pixel spans them
    float k = 1. - smoothstep(0.25, 1.5, pix / max(abs(dot(gn, rd)), 0.25));
    vec2 q = p.xz * 1.6;
    vec2 tilt = (vec2(vnoise(q), vnoise(q + 13.)) - 0.5) * 0.35 + (vec2(vnoise(q * 2.7 + 7.), vnoise(q * 2.7 + 21.)) - 0.5) * 0.18;
    vec3 wn = normalize(vec3(tilt.x * k, 1., tilt.y * k)), rr = reflect(rd, wn);
    rr.y = abs(rr.y);
    float fr = 0.02 + 0.98 * pow(1. - max(dot(-rd, wn), 0.), 5.);
    col = mix(col, skyBase(rr), fr * burn) + uSunCol * sh * burn * pow(max(dot(rr, L), 0.), 700.) * 40.;
  }
  return col + emi;
}

#if defined(SPLIT_P3) || defined(SUN_PASS)
// sun visibility of p against the mesh casters: the finest layer whose footprint holds p, four bilinear PCF taps.
// Layers hold every caster over their footprint, so meshes outside the view still shade what the view sees. The
// shading pass looks them up for each pixel, the sun cache for each 7 m cell.
float layerSun(int k, vec4 c) {
  vec3 s = c.xyz * 0.5 + 0.5;
  float z = min(s.z, 1.) - uShP[k].z, d = 0.5 * uShP[k].x;
  return 0.25 * (texture(tMeshSh, vec4(s.xy + vec2(-d, -d), float(k), z)) + texture(tMeshSh, vec4(s.xy + vec2(d, -d), float(k), z))
               + texture(tMeshSh, vec4(s.xy + vec2(-d, d), float(k), z)) + texture(tMeshSh, vec4(s.xy + vec2(d, d), float(k), z)));
}
float meshSun(vec3 p) {
  for (int k = uShN - 1; k >= 0; k--) {
    vec4 c = uShM[k] * vec4(p, 1.);
    if (max(abs(c.x), abs(c.y)) < 0.995) return layerSun(k, c);
  }
  return 1.;
}
#endif
#if defined(SPLIT_P3) && defined(MESH_TREES)
// sun visibility of p against the species the camera draws (the tree layer, template_c.html treeShadow)
float treeSun(vec3 p) {
  vec4 c = uShM[uShT] * vec4(p, 1.);
  return max(abs(c.x), abs(c.y)) < 0.995 ? layerSun(uShT, c) : 1.;
}
#endif

#ifdef TREE_PASS
// treeSD stands a crown on the ground under each of its points, so a crown reaching over a sheer face (a cliff's brink or
// foot) would fold down or up the face, sliced and hanging: no tree grows where the ground at its crown's edge lies more
// than TREE_DROP_M from its foot and, over some half of its radius, falls or climbs more steeply than TREE_SHEER (68
// degrees). A crown on a smooth bank is only sheared, and stays. A tree so dropped still counts in mh, which only has to
// bound the march.
#define TREE_DROP_M 4.
#define TREE_SHEER 2.5
bool treeStands(vec2 ctr, float r) {
  float foot = sampleGround(ctr).bed, drop = 0., steep = 0.;
  for (int k = ZERO; k < 8; k++) {
    float a = float(k) * 0.7853982;
    vec2 d = vec2(cos(a), sin(a));
    float mid = sampleGround(ctr + 0.5 * r * d).bed, edge = sampleGround(ctr + r * d).bed;
    drop = max(drop, abs(edge - foot));
    steep = max(steep, max(abs(mid - foot), abs(edge - mid)));
  }
  return drop <= TREE_DROP_M || steep <= TREE_SHEER * 0.5 * r;
}
void main() {                                         // one texel per tree cell: the table treeSD reads
  vec2 cc = floor(gl_FragCoord.xy) - TREE_G0;
  float d = 0., cf = 0., mh = 0., r;
  for (int k = ZERO; k < 25; k++) {                  // own cell (k == 12) and the tallest tree within two cells
    vec2 c = cc + vec2(float(k % 5), float(k / 5)) - 2.;
    vec4 h = hash42(c);
    vec2 ctr = (c + 0.25 + 0.5 * h.yz) * 7.;
    float conF, dn = floor(treeDens(ctr, conF) * 255. + 0.5) / 255.;   // as stored
    conF = floor(conF * 255. + 0.5) / 255.;
    if (k == 12) { d = dn; cf = conF; }
    if (h.x < dn) {
      mh = max(mh, treeH(h, dn, h.w < conF, r));
      if (k == 12 && !treeStands(ctr, r)) d = 0.;
    }
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
  int vmM = uDetail == 1 ? VM_TREES : 2;
  float t0 = 0.05;
  int ph = 0, gi = 0, cnt = 0, k = -1, kp = -1;      // ph 0 grid march, 1 baseline replay, 2 done
  if (ro.y > CEIL) { if (rd.y >= 0.) ph = 2; else t0 = (ro.y - CEIL) / -rd.y; }
  float t = t0, L = 2., Lo = 0., tr = t0, tlim = 0., tk = 0., tkp = 0., tIn = localEntry(ro, rd), tOut = treeTableExit(ro, rd);
  bool fresh = true;
  for (int i = ZERO; i < 530; i++) {                 // 160 grid steps + at most 360 replay samples
    if (ph == 0 && (t >= TMAX || gi >= 160)) { float tg = min(t, TMAX); tlim = tg - rk * tg; ph = 1; }
    if (ph == 1 && tr > tlim) break;
    if (ph == 2) break;
    bool rep = ph == 1;
    float ts = rep ? tr : t;
    vec3 p = ro + rd * ts;
    float dE = 1e5, dC = 1e5, dy = 1e5, vd = 1e5;
    if (rep || fresh) {                               // the single scene evaluation of this pass
      TS s = evalTop(p, ts, rep ? vmM : 0, tOut);
      dC = s.g.ed > 0. ? 1e5 : max(1. - s.g.ed, 1.);  // the local grid's edge: the grid march's clearance round p
      dy = p.y - topOf(s); vd = s.vd;
    }
    if (rep) {                                        // exactly trace()'s ST_MARCH advance
      tkp = tk; kp = k; tk = tr; k = cnt;
      float eps = min(0.00045 * tr + 0.01, 0.9);
      if (tr > TMAX || (p.y > CEIL && rd.y > 0.)) break;
      if (vd < eps || dy < 0.) { tk = tkp; k = kp; break; }   // keep the bisection bracket's start sample
      if (++cnt >= maxM) { k = -1; break; }          // baseline exhausts before tg: no skip for this block
      float km = mix(0.55, 0.85, smoothstep(900., 7000., tr));
      dE = tr < tIn ? max(tIn - tr + 1., 1.) : 1e5;   // up to where the ray enters the local grid, as trace() bounds it
      tr += min(min(max(km * dy, 0.0016 * tr + 0.03), max(dE, eps)), max(0.8 * vd, max(eps, 0.0012 * tr)));
      continue;
    }
    gi++;
    if (fresh) {                                      // clearance round the accepted point: the local grid's edge
      Lo = (0.75 * dC - rk * t - 1.) / (1. + rk);    // capsule within 0.75 dC - 1 m of p
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
  TS s = evalTop(vec3(xz.x, -1e4, xz.y), SUN_NEAR, 2, 0.);   // as the primary march sees it beyond SUN_NEAR (no micro-relief)
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
    TS q = evalTop(p, SUN_TREF, 0, 0.);
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
    TS s = evalTop(p, td, vm, 0.);
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
#ifdef MESH_TREES
  if (uShT >= 0 && o6.w >= 0.) shv.x = min(shv.x, treeSun(o6.xyz));
#endif
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
#elif defined(TINST_PASS)
// mesh trees: one texel per tree-table cell, for table rows uRow0 on: the tree treeSD's table path stands there (r = 0:
// none) and the ground it stands on. The page reads them back after each tree pass and keeps the trees as instances
uniform int uRow0;
void main() {
  ivec2 ic = ivec2(gl_FragCoord.xy) + ivec2(0, uRow0);
  vec2 cc = vec2(ic) - TREE_G0;
  vec4 h = hash42(cc);
  vec2 v = texelFetch(tTree, ic, 0).rg;               // density and conifer share, as treeSD reads them
  iOut0 = vec4(0.); iOut1 = vec4(0.); iOut2 = vec4(0.);
  if (h.x >= v.x) return;
  bool con = h.w < v.y;
  float r, H = treeH(h, v.x, con, r);
  vec2 ctr = (cc + 0.25 + 0.5 * h.yz) * 7.;
  // treeSD shears a tree to the ground under each of its points (bed + detailH). Here: the bed and the relief at the
  // centre (the relief as detailH has it from 700 m out, and its near octave apart) and their slopes across the crown,
  // kept apart because the relief fades out with distance
  float bed = 0., nF = 0., nN = 0.;
  vec4 sl = vec4(0.);                                 // d bed / dx, d bed / dz, d relief / dx, d relief / dz
  for (int k = ZERO; k < 5; k++) {
    vec2 o = k == 0 ? vec2(0.) : (k < 3 ? vec2(k == 1 ? r : -r, 0.) : vec2(0., k == 3 ? r : -r));
    GS g = sampleGround(ctr + o);
    float f = detailH(ctr + o, g, 1000.);
    if (k == 0) { bed = g.bed; nF = f; nN = detailH(ctr, g, 0.) - f; }
    else if (k < 3) sl.xz += vec2(g.bed, f) * ((k == 1 ? 0.5 : -0.5) / r);
    else sl.yw += vec2(g.bed, f) * ((k == 3 ? 0.5 : -0.5) / r);
  }
  iOut0 = vec4(ctr, r, H);
  iOut1 = vec4(bed, nF, nN, h.w + (con ? 2. : 0.));
  iOut2 = sl;
}
#elif defined(TREE_DRAW)
// mesh trees, stage A: each instance's crown and trunk, in the shapes treeSD gives them, ray cast within a screen quad
// round it (template_b.html treeVs). The depth pre-pass (the radial distance, at location 0) and the G-buffer pass (the
// record trace() writes for a 3D tree) run this one program, so their depths agree bit for bit
flat in vec4 vA;    // centre x, z, crown radius r, height H
flat in vec4 vB;    // base height, the ground's slope (x, z), hue (+2: conifer)
flat in float vF;   // the share of the tree this draws where it hands over to its mesh (TREE_MESH), by a dither; 1: all
//__TREE_DEPTH__
// the ray o + d s against a tree in its sheared frame (base at the origin, the ground's slope taken out): the nearest
// s > smin on its crown or trunk, with its part (1 crown, 2 trunk; 0 none), the shapes treeSD's distances bound
vec2 treeHit(vec3 o, vec3 d, float r, float H, bool con, float smin) {
  vec2 best = vec2(1e30, 0.);
  float rt = 0.25 + 0.012 * H, yt = con ? 0.5 * H : 0.64 * H;   // trunk: from 0.5 m down to its top inside the crown
  float A = dot(d.xz, d.xz), B = dot(o.xz, d.xz), C = dot(o.xz, o.xz) - rt * rt, D = B * B - A * C;
  if (D > 0. && A > 0.) {
    float q = sqrt(D);
    for (int i = 0; i < 2; i++) {
      float s = (-B + (i == 0 ? -q : q)) / A, y = o.y + d.y * s;
      if (s > smin && s < best.x && y >= -0.5 && y <= yt) best = vec2(s, 2.);
    }
  }
  if (con) {                                          // cone: a base disc of radius r at 0.12 H, its apex at H
    float k2 = r * r / (0.7744 * H * H), w0 = H - o.y;
    A = dot(d.xz, d.xz) - k2 * d.y * d.y; B = dot(o.xz, d.xz) + k2 * w0 * d.y; C = dot(o.xz, o.xz) - k2 * w0 * w0;
    D = B * B - A * C;
    if (D >= 0. && abs(A) > 1e-9) {
      float q = sqrt(D);
      for (int i = 0; i < 2; i++) {
        float s = (-B + (i == 0 ? -q : q)) / A, y = o.y + d.y * s;
        if (s > smin && s < best.x && y >= 0.12 * H && y <= H) best = vec2(s, 1.);
      }
    }
    if (abs(d.y) > 1e-9) {
      float s = (0.12 * H - o.y) / d.y;
      vec2 x = o.xz + d.xz * s;
      if (s > smin && s < best.x && dot(x, x) <= r * r) best = vec2(s, 1.);
    }
  } else {                                            // ellipsoid: semi-axes r, 0.36 H, r about 0.64 H up
    vec3 k = vec3(r, 0.36 * H, r), so = (o - vec3(0., 0.64 * H, 0.)) / k, sd = d / k;
    A = dot(sd, sd); B = dot(so, sd); C = dot(so, so) - 1.; D = B * B - A * C;
    if (D > 0.) {
      float q = sqrt(D);
      for (int i = 0; i < 2; i++) {
        float s = (-B + (i == 0 ? -q : q)) / A;
        if (s > smin && s < best.x) best = vec2(s, 1.);
      }
    }
  }
  return best;
}
void main() {
#ifdef TREE_REFL
  vec2 uv = (2. * (gl_FragCoord.xy * 2. + uJitter) - uRes) / uRes.y;   // the mirror is drawn at half resolution
#else
  vec2 uv = (2. * (gl_FragCoord.xy + uJitter) - uRes) / uRes.y;
#endif
  vec3 rd = normalize(uCamRot * vec3(uv * uTanHalf, 1.)), cam = uCamPos;
#ifdef TREE_REFL
  cam.y = 2. * LAKE - cam.y; rd.y = -rd.y;            // the mirror image on this ray is the tree on the mirrored ray
#endif
  if (vF < 1. && hash12(gl_FragCoord.xy + uJitter * 61.7) >= vF) discard;   // its mesh draws the rest (TREE_MESH)
  bool con = vB.w >= 2.;
  float H = vA.w;
  // the sheared frame, its origin moved along the ray to the nearest approach to the tree's middle, so the quadratics
  // stay well conditioned 5 km out
  vec3 o = cam - vec3(vA.x, vB.x, vA.y), d = rd;
  o.y -= dot(vB.yz, o.xz); d.y -= dot(vB.yz, d.xz);
  float tc = -dot(o - vec3(0., 0.5 * H, 0.), d) / dot(d, d);
  vec2 hit = treeHit(o + d * tc, d, vA.z, H, con, 0.05 - tc);
  float t = tc + hit.x;
  if (hit.y == 0. || ((uDbg & 2097152) == 0 && t >= TREE_FAR)) discard;   // the march's sheet from TREE_FAR on (debug: the table's edge)
  vec3 hp = cam + rd * t;
  vec2 dd = hp.xz - vA.xy;
#ifdef TREE_REFL
  // lit as reflFs lights the mirrored meshes, in shade()'s tree colours; the path length is the mirrored point's
  if (hp.y < LAKE - 0.05) discard;                   // under the water: not reflected
  vec3 q = o + d * t, n;                             // the hit in the tree's sheared frame
  if (hit.y > 1.5) n = vec3(q.x, 0., q.z);
  else if (!con) n = (q - vec3(0., 0.64 * H, 0.)) / vec3(vA.z * vA.z, 0.1296 * H * H, vA.z * vA.z);
  else n = q.y < 0.12 * H + 1e-3 ? vec3(0., -1., 0.) : vec3(q.x, vA.z * vA.z / (0.7744 * H * H) * (H - q.y), q.z);
  n = normalize(n + vec3(0., 1e-6, 0.));
  float hue = vB.w - (con ? 2. : 0.), hf = clamp((hp.y - vB.x) / max(H, 1.), 0., 1.);
  vec3 alb = hit.y > 1.5 ? vec3(0.05, 0.04, 0.03) : (con ? vec3(0.021, 0.037, 0.025) * (0.85 + 0.3 * hue)
                                                         : mix(vec3(0.034, 0.058, 0.016), vec3(0.06, 0.074, 0.02), hue));
  outColor = vec4(alb * (0.3 + 0.7 * hf) * (uSunCol * max(dot(n, uSunDir), 0.) + uSkyCol * (0.55 + 0.45 * n.y)), t);
#else
  bool shf = uSunDir.y > -0.03 && (uDbg & 1) == 0;   // trace()'s ST_SHINIT for a tree hit
  float sh = uSunDir.y < -0.03 ? 0. : 1.;
  if (shf && t >= SUN_NEAR && max(abs(hp.x), abs(hp.z)) < 5040.) { sh = sunCache(hp.xz); shf = false; }
  gOut0 = vec4(t, float(3 + (shf ? 64 : 0)), con ? 1. : 0., 0.);
  gOut1 = vec4(0., 1., 0., vB.x + dot(vB.yz, dd));   // the sheared ground under the hit, as treeSD's gy
  gOut2 = vA;
  gOut3 = vec4(con ? 1. : 0., con ? 0. : 0.36 * H, vB.w - (con ? 2. : 0.), hit.y);
  gOut4 = vec4(rd, 0.);
  gOut5 = vec4(0., 1., 0., 0.);
  gOut6 = shf ? vec4(hp + normalize(vec3(dd.x, max(length(dd), 0.5) * 0.6, dd.y)) + uSunDir * 0.05, 0.12 + t * 0.0015)
              : vec4(sh, 0., 0., -1.);
#endif
  gl_FragDepth = depthOf(t / length(vec3(uv * uTanHalf, 1.)));
}
#elif defined(TREE_MESH) || defined(TREE_SHADOW)
// mesh trees, stage B: the payload kit's species, a variant fitted to each tree near the camera (template_b.html
// treeMeshVs), drawn by the depth pre-pass and the G-buffer pass with the record a 3D tree writes, its part 3 (leaves)
// or 4 (bark) and its normal its own; shade() takes its species' colours. A leaf card is cut to leaves, and a tree
// handing over between two levels, or to its drawn crown (TREE_DRAW), is dithered between them alike in both passes
in vec3 vP; in vec3 vN; in vec2 vUV; in float vAux; in float vGy;
in vec2 vC;          // across a leaf card, 0 to 1
flat in uint vPart;  // 0 bark, 1 solid leaves, 2 a leaf card
flat in vec4 vA;     // centre x, z, crown radius r, height H
flat in vec4 vB;     // base height, the ground's slope (x, z), hue (+2: conifer)
flat in vec4 vI;     // species, the farther level's share where two hand over, this level the farther (1) or not, unused
float leafMask(vec2 uv) {                             // a leaf card's leaves: one round-tipped leaf in each cell of a 3.5 grid
  vec2 g = uv * 3.5, c = floor(g);
  float m = -1.;
  for (int j = ZERO; j < 9; j++) {
    vec2 k = c + vec2(float(j % 3 - 1), float(j / 3 - 1));
    vec4 h = hash42(k + 31.7);
    vec2 q = g - k - 0.2 - 0.6 * h.xy;
    float a = 6.2832 * h.z, ca = cos(a), sa = sin(a);
    m = max(m, 1. - length(vec2(ca * q.x + sa * q.y, ca * q.y - sa * q.x) / vec2(0.62, 0.3)));
  }
  return m;
}
// how far a leaf card's leaves shrink where it is drawn: from CARD_EDGE (0.6) of its half-side out from its centre to
// none at the circle its edges touch (mesh_trees card_fill), so a card reads as a clump of leaves, not a square
float leafEdge() { return smoothstep(0.6, 1., length(vC * 2. - 1.)); }
#ifdef TREE_SHADOW
// the tree layer's depth (template_c.html treeShadow): a leaf card cut to its leaves at their own size, so that a
// shadow keeps its leaves' cover whatever the layer's texel
void main() { if (vPart == 2u && leafMask(vUV) < leafEdge()) discard; }
#else
// a leaf card's leaves widen as they shrink under three pixels across, to half again their size under one (debug bit
// 8388608: never): a far card keeps its cover rather than breaking into specks between the pixels' centres
float leafGrow(vec2 uv) {
  vec2 w = fwidth(uv) * 3.5;                          // leafMask's grid cells per pixel; a leaf is 0.6 of a cell across
  return (uDbg & 8388608) != 0 ? 0. : 0.5 * smoothstep(0.2, 0.6, max(w.x, w.y));
}
// how far along d the ray from o (in the tree's sheared frame) is inside the tree as treeSD has it, before it leaves:
// its crown or its trunk, or from bark its trunk only (0 if it never is)
float treeExit(vec3 o, vec3 d, float r, float H, bool con, bool bark) {
  float s = 0., rt = 0.25 + 0.012 * H, yt = con ? 0.5 * H : 0.64 * H;
  float A = dot(d.xz, d.xz), B = dot(o.xz, d.xz), C = dot(o.xz, o.xz) - rt * rt, D = B * B - A * C;
  if (D > 0. && A > 0.) { float e = (-B + sqrt(D)) / A, y = o.y + d.y * e; if (y >= -0.5 && y <= yt) s = e; }
  if (bark) return max(s, 0.);
  if (con) {                                          // the cone between its base disc at 0.12 H and its apex at H
    float k2 = r * r / (0.7744 * H * H), w0 = H - o.y;
    A = dot(d.xz, d.xz) - k2 * d.y * d.y; B = dot(o.xz, d.xz) + k2 * w0 * d.y; C = dot(o.xz, o.xz) - k2 * w0 * w0;
    D = B * B - A * C;
    if (D >= 0. && abs(A) > 1e-9) for (int i = 0; i < 2; i++) {
      float e = (-B + (i == 0 ? -1. : 1.) * sqrt(D)) / A, y = o.y + d.y * e;
      if (y >= 0.12 * H && y <= H) s = max(s, e);
    }
    if (abs(d.y) > 1e-9) { float e = (0.12 * H - o.y) / d.y; vec2 x = o.xz + d.xz * e; if (dot(x, x) <= r * r) s = max(s, e); }
  } else {                                            // the ellipsoid, semi-axes r, 0.36 H, r about 0.64 H up
    vec3 k = vec3(r, 0.36 * H, r), so = (o - vec3(0., 0.64 * H, 0.)) / k, sd = d / k;
    A = dot(sd, sd); B = dot(so, sd); C = dot(so, so) - 1.; D = B * B - A * C;
    if (D > 0.) s = max(s, (-B + sqrt(D)) / A);
  }
  return max(s, 0.);
}
void main() {
  float grow = leafGrow(vUV);                         // before any discard: derivatives need the whole quad
  if (vI.y > 0. && (vI.z < 0.5) == (hash12(gl_FragCoord.xy + uJitter * 61.7) < vI.y)) discard;   // the other level's share
  float edge = (uDbg & 16777216) != 0 ? 0. : leafEdge();   // debug bit 16777216: the cards cut square
  if (vPart == 2u && leafMask(vUV) < edge - grow * (1. - edge)) discard;
  vec3 d = vP - uCamPos; float t = length(d);
  vec3 n = normalize(vN);
  if (vPart == 0u && dot(n, d) > 0.) n = -n;        // bark seen from behind (a limb's open end): its near side
  bool con = vB.w >= 2., shf = uSunDir.y > -0.03 && (uDbg & 1) == 0;   // trace()'s ST_SHINIT for a tree hit
  float sh = uSunDir.y < -0.03 ? 0. : 1.;
  if (shf && t >= SUN_NEAR && max(abs(vP.x), abs(vP.z)) < 5040.) { sh = sunCache(vP.xz); shf = false; }
  // the sun's march starts where its ray leaves this tree as the march sees it, so the tree never shades itself through
  // the shapes it no longer has: its normals and its leaves' depth in the crown (aux) shade it instead
  vec3 o = vP - vec3(vA.x, vB.x, vA.y), sd = uSunDir;
  o.y -= dot(vB.yz, o.xz); sd.y -= dot(vB.yz, sd.xz);
  vec3 so = vP + uSunDir * (treeExit(o, sd, vA.z, vA.w, con, vPart == 0u) + 0.05);
  gOut0 = vec4(t, float(3 + (shf ? 64 : 0)), vAux, 0.);
  gOut1 = vec4(n, vGy);
  gOut2 = vA;
  gOut3 = vec4(con ? 1. : 0., vI.x, vB.w - (con ? 2. : 0.), vPart == 0u ? 4. : 3.);
  gOut4 = vec4(d / t, 0.);
  gOut5 = vec4(0., 1., 0., 0.);
  gOut6 = shf ? vec4(so, 0.12 + t * 0.0015) : vec4(sh, 0., 0., -1.);
}
#endif
#endif
