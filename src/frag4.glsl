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

// surface classes of the 2 m layer (from the essay's grounds/Hogsmeade map colours)
vec3 classCol(float c, vec2 xz) {
  if (c < 0.5) return vec3(0.075, 0.092, 0.034);                                   // rough meadow
  if (c < 1.5) return vec3(0.055, 0.105, 0.028);                                   // lawn
  if (c < 2.5) return mix(vec3(0.085, 0.045, 0.075), vec3(0.065, 0.058, 0.032), vnoise(xz * 0.35)); // heather in bloom
  if (c < 3.5) return vec3(0.16, 0.13, 0.09);                                      // earth track
  if (c < 4.5) return mix(vec3(0.12, 0.085, 0.045), vec3(0.06, 0.08, 0.025), step(0.5, fract(xz.x * 0.4))); // vegetable beds
  if (c < 5.5) return vec3(0.045, 0.04, 0.025);                                    // forest floor
  if (c < 6.5) return vec3(0.048, 0.115, 0.028);                                   // Quidditch pitch
  return vec3(0.12, 0.115, 0.1);                                                    // shingle
}

vec3 groundAlb(vec3 p, TS s, float t, float pix, vec3 n) {
  GS g = s.g;
  vec3 sat = satAlb(p.xz, pix);
  vec3 a = sat;
  vec2 fuv = vec2(dot(uForU, vec3(p.xz, 1.)), dot(uForV, vec3(p.xz, 1.)));
  vec2 fe = min(fuv, 1. - fuv) * uForM;
  float fw = smoothstep(0., 300., min(fe.x, fe.y));
  if (fw > 0.) a = mix(a, vec3(0.07, 0.1, 0.034), textureLod(tFor, fuv, 0.).g * fw * 0.6);
  if (g.lw > 0.) {
    vec2 j = (vec2(vnoise(p.xz * 0.6), vnoise(p.xz * 0.6 + 9.)) - 0.5) * 2.4;
    vec2 luv = vec2(dot(uLocU, vec3(p.xz + j, 1.)), dot(uLocV, vec3(p.xz + j, 1.)));
    float c = floor(textureLod(tLocN, luv, 0.).r * 255. + 0.5);
    vec3 cc = classCol(c, p.xz);
    if (c > 5.5 && c < 6.5) { vec2 u = pitchUV(p.xz); cc *= 0.86 + 0.28 * step(0.5, fract(u.x / 9.)); }
    float lum = dot(sat, vec3(0.3, 0.55, 0.15));
    cc *= mix(1., clamp(lum / 0.075, 0.65, 1.45), 0.3);
    a = mix(a, cc, g.lw);
    a = mix(a, vec3(0.17, 0.145, 0.11) * (0.85 + 0.3 * vnoise(p.xz * 1.3)), g.la.r * g.lw);
  }
  a *= mix(1., 0.6, s.dens);
  a *= mix(0.82 + 0.36 * vnoise(p.xz * 0.35), 1., smoothstep(600., 1500., t));
  float rock = smoothstep(0.72, 0.42, n.y) * (1. - 0.5 * g.lw * smoothstep(0.42, 0.6, n.y));
  vec3 rc = vec3(0.1, 0.094, 0.084) * (0.7 + 0.6 * vnoise(p.xz * 0.11 + p.y * 0.07))
          * (0.86 + 0.14 * sin(p.y * 1.9 + 3. * vnoise(p.xz * 0.05)))           // weathered strata
          * (0.65 + 0.6 * vnoise(vec2((p.x + p.z) * 0.35, p.y * 0.03)));          // vertical gullies and streaks
  rc = mix(rc, vec3(0.05, 0.065, 0.03), smoothstep(0.55, 0.8, vnoise(p.xz * 0.3 + p.y * 0.2)) * 0.6);   // moss
  return mix(a, rc, rock * 0.9);
}

// ------------------------------------------------------------------ sky, clouds, haze
#define CLOUD_Y 2300.
float cloudDen(vec2 xz) {
  vec2 q = xz * 0.00028 + uTime * vec2(0.0035, 0.0012);
  return smoothstep(0.5, 0.82, fbm4(q) + 0.12 * vnoise(q * 6.));
}
float cloudShadow(vec3 p) {
  if (uSunDir.y < 0.03) return 1.;
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
    c += vec3(0.6, 0.65, 0.8) * step(0.9975, hash12(floor(sc))) * uDusk * smoothstep(0.02, 0.2, rd.y) * 0.5;
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
  float fa = 1. - exp(-FOG_DENS * exp(-b * (ro.y - 150.)) * path);
  float sd = max(dot(rd, uSunDir), 0.);
  vec3 fc = mix(uHorCol * 0.95, uSunCol * 0.75 + uHorCol * 0.45, pow(sd, 5.) * 0.65)
          + uGlow * 0.6 * pow(max(dot(normalize(rd.xz + 1e-5), normalize(uSunDir.xz + 1e-5)), 0.), 4.);
  return mix(col, fc, fa);
}

// ------------------------------------------------------------------ windows (lit at dusk)
float windowMask(vec3 p, float wx, float y0, float y1, vec2 cs, float pix, out float lit) {
  vec2 w = vec2(wx, p.y) / cs;
  vec2 c = floor(w), f = fract(w);
  lit = step(0.6, hash12(c + 31.7));
  float m = step(0.34, f.x) * step(f.x, 0.66) * step(0.26, f.y) * step(f.y, 0.8);
  float aa = smoothstep(0.2, 0.7, pix / min(cs.x, cs.y));      // sub-pixel windows -> their average coverage
  m = mix(m, 0.17, aa); lit = mix(lit, 0.14, aa);             // biased dark: a distant lit castle reads as a dark mass, not a lamp
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
