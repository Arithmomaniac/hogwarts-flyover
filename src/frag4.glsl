// ------------------------------------------------------------------ albedo
float lodFor(float pix, float texel) { return log2(max(pix / texel, 1.)); }

vec3 satAlb(vec2 xz, float pix) {
  float m = max(abs(xz.x), abs(xz.y));
  vec3 c;
  if (m < 4700.) c = textureLod(tMS, uvM(xz), lodFor(pix, 10.)).rgb;
  else if (m < 11600.) {
    c = textureLod(tAS, uvA(xz), lodFor(pix, 30.)).rgb;
    if (m < 4990.) c = mix(textureLod(tMS, uvM(xz), lodFor(pix, 10.)).rgb, c, smoothstep(4700., 4990., m));
  } else {
    c = textureLod(tBS, uvB(xz), lodFor(pix, 120.)).rgb;
    if (m < 11990.) c = mix(textureLod(tAS, uvA(xz), lodFor(pix, 30.)).rgb, c, smoothstep(11600., 11990., m));
  }
  return c * SATGAIN;
}

// the vegetable beds: 1.2 m beds of crops between narrow paths of trodden soil and grass, plot by plot (6 m by 11 m) a
// different crop or dug-over soil; the crops' rows fade where a pixel spans more than a few of them, and the beds where
// it spans more than one
vec3 bedsCol(vec2 xz, float pix) {
  vec2 q = vec2(dot(xz, vec2(0.906, 0.423)), dot(xz, vec2(-0.423, 0.906)));       // across the beds, along them
  float crop = hash12(floor(q / vec2(6., 11.)) + 0.5);
  vec3 soil = mix(vec3(0.112, 0.082, 0.046), vec3(0.086, 0.065, 0.039), vnoise(xz * 1.7));
  vec3 leaf = mix(vec3(0.046, 0.078, 0.024), vec3(0.078, 0.096, 0.034), fract(crop * 7.)) * (0.75 + 0.5 * vnoise(xz * 2.3 + crop * 37.));
  vec2 d = abs(fract(q / vec2(1.5, 11.)) - 0.5) * vec2(1.5, 11.);                  // metres from a bed's middle, a plot's
  float far = smoothstep(0.3, 1.2, pix);
  float bed = mix(smoothstep(0.66, 0.54, d.x) * smoothstep(5.45, 5.2, d.y), 0.8, far);
  float rows = mix(0.55 + 0.45 * sin(q.x * 15.7), 0.75, smoothstep(0.06, 0.2, pix));
  vec3 grown = mix(soil, leaf, smoothstep(0.15, 0.3, crop) * rows);               // a fifth of the plots dug over
  return mix(mix(soil, vec3(0.075, 0.08, 0.04), 0.6), grown, bed);
}

// surface classes of the 2 m layer (from the essay's grounds/Hogsmeade map colours), pix the pixel's footprint (m)
vec3 classCol(float c, vec2 xz, float pix) {
  if (c < 0.5) return vec3(0.075, 0.092, 0.034);                                   // rough meadow
  if (c < 1.5) return vec3(0.055, 0.105, 0.028);                                   // lawn
  if (c < 2.5) return mix(vec3(0.066, 0.074, 0.031), vec3(0.082, 0.068, 0.036), vnoise(xz * 0.35)) * (0.8 + 0.4 * vnoise(xz * 0.6 + 3.)); // rough scrubland
  if (c < 3.5) return vec3(0.16, 0.13, 0.09);                                      // earth track
  if (c < 4.5) return bedsCol(xz, pix);                                            // vegetable beds
  if (c < 5.5) return mix(mix(vec3(0.054, 0.045, 0.027), vec3(0.046, 0.038, 0.024), vnoise(xz * 1.3 + 3.)),  // forest floor:
                          vec3(0.046, 0.06, 0.026), smoothstep(0.4, 0.75, vnoise(xz * 0.28)))   // leaf and needle litter,
                   * (0.82 + 0.36 * vnoise(xz * 0.9 + 7.));                        // moss in patches, twigs and hollows
  if (c < 6.5) return vec3(0.048, 0.115, 0.028);                                   // Quidditch pitch
  if (c < 7.5) return vec3(0.12, 0.115, 0.1);                                      // shingle
  return mix(vec3(0.058, 0.068, 0.029), vec3(0.07, 0.062, 0.034), vnoise(xz * 0.21))  // overgrown garden: rank grass and
    * (0.8 + 0.4 * vnoise(xz * 0.8 + 5.));                                          // dead stems, nettle and bramble clumps
}

vec3 surfCol(float c, vec2 xz, float pix) {          // a class's colour, the pitch's mown stripes included
  vec3 cc = classCol(c, xz, pix);
  if (c > 5.5 && c < 6.5) cc *= 0.86 + 0.28 * step(0.5, fract(pitchUV(xz).x / 9.));
  return cc;
}
float classAt(ivec2 i, ivec2 n) { return floor(texelFetch(tLocB, clamp(i, ivec2(0), n - 1), 0).b * 255. + 0.5); }
bool hardClass(float c) { return c > 2.5 && c < 6.5 && abs(c - 5.) > 0.5; }    // paths, beds and the pitch
// class c's share of a point by the cubic B-spline over the 4 x 4 texels round it (i: the cell's first texel, w: the
// point's place in it). The beds and the pitch take a point where it is a half or more, so their edges run smooth where
// the texels step slantwise (read as the four nearest texels weigh them, the beds' edge below the greenhouses traced the
// texels' steps), and a lone texel (0.44 at most) claims no ground. Paths keep the four nearest texels' weight: a path a
// texel wide running slantwise reaches a half by the B-spline only at its texels' centres, so it would vanish
float splineShare(ivec2 i, vec2 w, ivec2 n, float c) {
  vec2 v = 1. - w;
  vec4 bx = vec4(v.x * v.x * v.x, 4. - 6. * w.x * w.x + 3. * w.x * w.x * w.x, 4. - 6. * v.x * v.x + 3. * v.x * v.x * v.x, w.x * w.x * w.x) / 6.;
  vec4 by = vec4(v.y * v.y * v.y, 4. - 6. * w.y * w.y + 3. * w.y * w.y * w.y, 4. - 6. * v.y * v.y + 3. * v.y * v.y * v.y, w.y * w.y * w.y) / 6.;
  float s = 0.;
  for (int y = 0; y < 4; y++)
    for (int x = 0; x < 4; x++) s += bx[x] * by[y] * float(classAt(i + ivec2(x - 1, y - 1), n) == c);
  return s;
}

// the colours of the four texels' classes c (8 and up: the overgrown garden), each texel weighed by k. surfCol is called
// once for each class, on a constant, which the HLSL compiler folds to that class's colour alone; a call on a texel's
// class inlines every branch of classCol, and three of them made Edge's compile of the shading pass take a minute and a
// half (REBUILD.md). Written out class by class: ANGLE asks the compiler to keep loops, so a loop's call is not folded
vec3 classTerm(vec3 sum, vec4 c, vec4 k, float j, vec2 xz, float pix) {
  float w = dot(k, vec4(equal(c, vec4(j))));
  if (w > 0.) sum += w * surfCol(j, xz, pix);
  return sum;
}
vec3 classesCol(vec4 c, vec4 k, vec2 xz, float pix) {
  c = min(c, vec4(8.));
  vec3 s = classTerm(vec3(0.), c, k, 0., xz, pix);
  s = classTerm(s, c, k, 1., xz, pix);
  s = classTerm(s, c, k, 2., xz, pix);
  s = classTerm(s, c, k, 3., xz, pix);
  s = classTerm(s, c, k, 4., xz, pix);
  s = classTerm(s, c, k, 5., xz, pix);
  s = classTerm(s, c, k, 6., xz, pix);
  s = classTerm(s, c, k, 7., xz, pix);
  return classTerm(s, c, k, 8., xz, pix);
}

// the ground's colour from the surface classes of the four texels round luv (the local layer's uv), as bilinear
// filtering weighs them from their centres: soft classes blend across the texel between them, while a hard class keeps a
// sharp edge, taking the point wherever its weight is a half or more (the beds and the pitch: their B-spline share)
vec3 classMix(vec2 luv, vec2 xz, float pix) {
  ivec2 n = textureSize(tLocB, 0);
  vec2 f = luv * vec2(n) - 0.5, w = fract(f);
  ivec2 i = ivec2(floor(f));
  vec4 c = vec4(classAt(i, n), classAt(i + ivec2(1, 0), n), classAt(i + ivec2(0, 1), n), classAt(i + ivec2(1, 1), n));
  vec4 k = vec4(1., 0., 0., 0.);                   // the texels' weights in the colour, over soft
  float soft = 1.;
  if (!all(equal(c, c.xxxx))) {
    vec4 wt = vec4((1. - w.x) * (1. - w.y), w.x * (1. - w.y), (1. - w.x) * w.y, w.x * w.y);
    vec4 hk = vec4(hardClass(c.x), hardClass(c.y), hardClass(c.z), hardClass(c.w));
    float hard = -1., most = 0.;
    for (int j = 0; j < 4; j++) {
      float own = dot(wt, vec4(equal(c, vec4(c[j]))));
      if (hardClass(c[j]) && own > most) { most = own; hard = c[j]; }
    }
    k = wt * (1. - hk);
    soft = k.x + k.y + k.z + k.w;
    if (soft <= 0. || (hard >= 0. && (hard < 3.5 ? most : splineShare(i, w, n, hard)) >= 0.5)) {
      c = vec4(hard);
      k = vec4(1., 0., 0., 0.);
      soft = 1.;
    }
  }
  return classesCol(c, k, xz, pix) / soft;
}

// rock on an upright plane, (along it, up): weathering over tens of metres, beds dipping along it and joints leaning up
// it, each wandering at its own scale, and grain, on coordinates warped so that no lattice shows
float rockTex(vec2 q) {
  q += 3. * vec2(vnoise(q * 0.07 + 3.), vnoise(q * 0.07 + 11.)) - 1.5;
  vec2 b = mat2(0.96, -0.28, 0.28, 0.96) * q, j = mat2(0.96, 0.28, -0.28, 0.96) * q;
  b.y += 1.4 * vnoise(q * 0.35 + 23.);
  j.x += 1.6 * vnoise(q * 0.3 + 29.);
  return (0.72 + 0.56 * vnoise(q * 0.09))
       * (0.82 + 0.3 * vnoise(vec2(b.x * 0.08, b.y * 0.55) + 7.))
       * (0.82 + 0.3 * vnoise(vec2(j.x * 0.45, j.y * 0.07) + 13.))
       * (0.85 + 0.3 * vnoise(q * 1.3 + 17.));
}
// rock on the two upright planes, blended by the face's normal, so a face of any bearing shows it unstretched; moss.
// The meshes that stand in for cut cliff share it (the harbour mouth's rock, set piece 34)
vec3 rockAlb(vec3 p, vec3 n) {
  vec2 w = pow(abs(n.xz) + 1e-3, vec2(4.));
  float k = w.x / (w.x + w.y);
  float a = k > 0.97 ? 0. : rockTex(p.xy + 31.), b = k < 0.03 ? 0. : rockTex(p.zy);  // a call site each (see classesCol)
  float r = k > 0.97 ? b : (k < 0.03 ? a : mix(a, b, k));
  return mix(vec3(0.088, 0.082, 0.073) * r, vec3(0.05, 0.065, 0.03), smoothstep(0.55, 0.8, vnoise(p.xz * 0.3 + p.y * 0.2)) * 0.6);
}
// the rock's bumps on faces steeper than the ground's (shared with set piece 34, so the two shade alike), fading out
// where a pixel spans them: value noise on two lattices turned 41 degrees and scaled 1.37 times against each other
// (0.64: 0.9 over root 2, the spread of one), so that neither lattice's rows show. On one, the cliff's rock repeated
// every 2 m along its face
vec3 rockBump(vec3 p, vec3 n, float pix) {
  float rk = smoothstep(0.72, 0.42, n.y) * (1. - smoothstep(0.3, 1.5, pix));
  if (rk <= 0.) return n;
  vec3 q = vec3(1.04 * p.x - 0.89 * p.z + 3.1, 1.37 * p.y + 5.7, 0.89 * p.x + 1.04 * p.z + 7.9);
  vec3 d = vec3(vnoise(p.xz * 0.45 + p.y * 0.3), vnoise(p.zy * 0.45 + 5.), vnoise(p.xy * 0.45 + 9.))
         + vec3(vnoise(q.xz * 0.45 + q.y * 0.3), vnoise(q.zy * 0.45 + 5.), vnoise(q.xy * 0.45 + 9.)) - 1.;
  return normalize(n + d * 0.64 * rk);
}

vec3 groundAlb(vec3 p, TS s, float t, float pix, vec3 n, float shingle) {
  GS g = s.g;
  vec3 sat = satAlb(p.xz, pix);
  vec3 a = sat;
  vec2 fuv = vec2(dot(uForU, vec3(p.xz, 1.)), dot(uForV, vec3(p.xz, 1.)));
  vec2 fe = min(fuv, 1. - fuv) * uForM;
  float fw = smoothstep(0., 300., min(fe.x, fe.y));
  if (fw > 0.) a = mix(a, vec3(0.07, 0.1, 0.034), textureLod(tFor, fuv, 0.).g * fw * 0.6);
  if (g.lw > 0.) {
    vec2 j = (vec2(vnoise(p.xz * 0.6), vnoise(p.xz * 0.6 + 9.)) - 0.5) * 2.4;
    if ((uDbg & 2048) != 0) j = vec2(0.);           // debug: class edges without their noise
    vec2 luv = vec2(dot(uLocU, vec3(p.xz + j, 1.)), dot(uLocV, vec3(p.xz + j, 1.)));
    vec3 cc = (uDbg & 4) != 0                        // debug: the class of the nearest texel alone
      ? surfCol(classAt(ivec2(floor(luv * vec2(textureSize(tLocB, 0)))), textureSize(tLocB, 0)), p.xz, pix)
      : classMix(luv, p.xz, pix);
    // the lake's shingle from its smooth field (the page blurs its texels at load), the edge broken by the same noise
    cc = mix(cc, classCol(7., p.xz, pix), smoothstep(0.32, 0.48, shingle + 0.125 * j.x));
    float lum = dot(sat, vec3(0.3, 0.55, 0.15));
    cc *= mix(1., clamp(lum / 0.075, 0.65, 1.45), 0.3);
    a = mix(a, cc, g.lw);
    a = mix(a, vec3(0.17, 0.145, 0.11) * (0.85 + 0.3 * vnoise(p.xz * 1.3)), g.la.r * g.lw);
  }
  a *= mix(1., 0.6, s.dens);
  a *= mix(0.82 + 0.36 * vnoise(p.xz * 0.35), 1., smoothstep(600., 1500., t));
  float rock = smoothstep(0.72, 0.42, n.y) * (1. - 0.5 * g.lw * smoothstep(0.42, 0.6, n.y));
  return rock > 0. ? mix(a, rockAlb(p, n), rock * 0.9) : a;
}

// ------------------------------------------------------------------ sky, clouds, haze
#define CLOUD_Y 2300.
float cloudDen(vec2 xz) {
  vec2 q = xz * 0.00028 + uTime * vec2(0.0035, 0.0012);
  return smoothstep(0.5, 0.82, fbm4(q) + 0.12 * vnoise(q * 6.));
}
float cloudShadow(vec3 p) {
  if (uSunDir.y < 0.03 || (uDbg & 1024) != 0) return 1.;   // debug: the ground lit as under a clear sky
  return 1. - 0.65 * cloudDen(p.xz + uSunDir.xz * (CLOUD_Y - p.y) / uSunDir.y);
}

vec3 skyBase(vec3 rd) {
  vec3 c = mix(uHorCol, uSkyCol, 1. - exp(-max(rd.y, 0.) * 7.));
  if (rd.y < 0.) c = mix(uHorCol, uHorCol * 0.6, clamp(-rd.y * 6., 0., 1.));
  float sd = max(dot(rd, uSunDir), 0.);
  float az = max(dot(normalize(rd.xz + 1e-5), normalize(uSunDir.xz + 1e-5)), 0.);
  c += uGlow * pow(az, 4.) * exp(-max(rd.y, 0.) * 9.);
  return c + uSunCol * (0.05 * pow(sd, 4.) + 0.22 * pow(sd, 32.) + 0.5 * pow(sd, 300.));
}

vec3 sky(vec3 ro, vec3 rd) {
  vec3 c = skyBase(rd);
  float sd = max(dot(rd, uSunDir), 0.);
  c += uSunCol * 18. * smoothstep(0.99985, 0.99995, sd) * step(0., rd.y);
  if (rd.y > 0.) {
    vec2 sc = rd.xz / (rd.y + 1.) * 520.;
    // stars come out as the twilight fades, from the sun 3 degrees down to about 11 (none in a civil-dusk sky)
    c += vec3(0.6, 0.65, 0.8) * step(0.9975, hash12(floor(sc))) * (1. - smoothstep(STARS_SUN_Y.x, STARS_SUN_Y.y, uSunDir.y))
       * smoothstep(0.02, 0.2, rd.y) * 0.5;
  }
  if (rd.y > 0.004) {
    float tc = (CLOUD_Y - ro.y) / rd.y;
    if (tc > 0.) {
      vec2 pc = ro.xz + rd.xz * tc;
      float d = cloudDen(pc);
      if (d > 0.) {
        float d2 = cloudDen(pc + uSunDir.xz / max(uSunDir.y, 0.08) * 120.);
        vec3 cc = mix(uSkyCol * 0.5 + uHorCol * 0.35, uSunCol * 1.05 + uSkyCol * 0.45, exp(-2.2 * d2));
        cc += uSunCol * pow(sd, 6.) * 0.8 * (1. - d);
        c = mix(c, cc, d * 0.92 * exp(-tc * 0.000032));
      }
    }
  }
  return c;
}

vec3 applyFog(vec3 col, vec3 ro, vec3 rd, float t) {
  float b = 1. / 1100.;
  float k = b * rd.y * t;
  float path = abs(k) < 1e-3 ? t : (1. - exp(-k)) / (b * rd.y);
  float fa = 1. - exp(-FOG_DENS * exp(-b * (ro.y - (LAKE - 3.))) * path);
  float sd = max(dot(rd, uSunDir), 0.);
  vec3 fc = mix(uHorCol * 0.95, uSunCol * 0.75 + uHorCol * 0.45, pow(sd, 5.) * 0.65)
          + uGlow * 0.6 * pow(max(dot(normalize(rd.xz + 1e-5), normalize(uSunDir.xz + 1e-5)), 0.), 4.);
  return mix(col, fc, fa);
}

// ------------------------------------------------------------------ windows (lit at dusk)
float windowMask(vec3 p, float wx, float y0, float y1, vec2 cs, float pix, out float lit) {
  vec2 w = vec2(wx, p.y) / cs;
  vec2 c = floor(w), f = fract(w);
  lit = step(0.6, hash12(c + 31.7)) * (0.5 + 0.5 * hash12(c + 7.31));   // two in five lit, each room its own brightness
  float m = step(0.34, f.x) * step(f.x, 0.66) * step(0.26, f.y) * step(f.y, 0.8);
  float aa = smoothstep(0.2, 0.7, pix / min(cs.x, cs.y));      // sub-pixel windows -> their average coverage
  m = mix(m, 0.17, aa); lit = mix(lit, 0.1, aa);              // biased dark: a distant lit castle reads as a dark mass, not a lamp
  return m * step(y0, p.y) * step(p.y, y1);
}
// horizontal window coordinate along a wall; snap = align to the essay map grid (25.1 deg) and its diagonals,
// which the castle and village walls follow, so noisy texture normals don't scramble the window rows
float faceCoord(vec3 p, vec3 n, bool snap) {
  float a = atan(n.z, n.x);
  if (snap) a = floor((a - 0.4381) / 0.7854 + 0.5) * 0.7854 + 0.4381;
  return dot(p.xz, vec2(-sin(a), cos(a)));
}

// ------------------------------------------------------------------ house colours for the stands
vec3 standCol(vec2 xz) {
  vec2 u = pitchUV(xz) / vec2(101.6, 54.6);
  float s = floor(mod(atan(u.y, u.x) / (PI * 0.5) + 4.5, 4.));
  return s < 0.5 ? vec3(0.4, 0.03, 0.02) : (s < 1.5 ? vec3(0.03, 0.05, 0.24) : (s < 2.5 ? vec3(0.03, 0.16, 0.05) : vec3(0.48, 0.34, 0.03)));
}
