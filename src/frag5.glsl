// ------------------------------------------------------------------ shading
vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0., 1.); }

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
  if (r.mat == 1 || (r.mat == 2 && depth < 4.)) galb = groundAlb(p, s, t, pix / max(abs(dot(gn, rd)), 0.25), gn);

  if (r.mat == 2) {                                   // loch / river: peaty water, Fresnel reflection, sun glint
    float fr = (0.02 + 0.98 * pow(1. - max(dot(-rd, r.wn), 0.), 5.)) * smoothstep(0., 0.6, depth);   // thin shore water: less mirror
    vec3 refl = skyc;
    if (r.rmat != 0) {
      vec3 q = r.rp, rc;
      if (r.rmat == 4) {
        vec3 n = normalize(vec3(-r.rd.x, 0., -r.rd.z));
        rc = vec3(0.24, 0.225, 0.2) * (uSunCol * max(dot(n, L), 0.) + uSkyCol * 0.55);
        float lit; float wm = windowMask(q, faceCoord(q, n, true), 192., r.rctop - 1.5, vec2(3.2, 4.1), (t + r.rt) * 2. * uTanHalf / uRes.y, lit);
        rc += wm * lit * uDusk * vec3(1., 0.6, 0.25) * 0.9;
      } else {
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
      float ss = textureLod(tLocB, s.g.luv, 0.).g * 25.5 - 12.8;
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
    alb = s.tp.w < 0. ? mix(vec3(0.045, 0.064, 0.018), con, s.conF) : (s.tp.x > 0.5 ? con : dec);
    ao = s.tp.w < 0. ? 0.7 : 0.3 + 0.7 * clamp((p.y - s.gy) / max(s.tc.w, 1.), 0., 1.);
    wrap = 0.3;
  } else if (r.mat == 4) {                            // castle
    float top = r.ctop, rt = r.aux > 99.5 ? -1. : r.aux;
    int ti = r.aux > 99.5 ? int(r.aux - 100. + 0.5) - 1 : -1;
    vec2 bw = vec2(faceCoord(p, n, true) / 1.3, p.y / 0.62);
    bw.x += 0.5 * mod(floor(bw.y), 2.);
    alb = vec3(0.25, 0.235, 0.21) * (0.86 + 0.28 * hash12(floor(bw)) * (1. - smoothstep(0.15, 0.6, pix)));
    alb *= 0.88 + 0.24 * vnoise(p.xz * 0.05 + p.y * 0.04);
    bool roof = false; float wx = faceCoord(p, n, true), wtop = top - 1.5;
    if (ti >= 0) {
      vec4 tw = uTw[ti]; float rf = uTx[ti].x;
      vec2 q = p.xz - tw.xy;
      wx = atan(q.y, q.x) * tw.z; wtop = tw.w - 1.5;
      if (rf > 3.5 && rf < 4.5 && p.y > tw.w + 0.6) { alb = vec3(0.05, 0.085, 0.075); spec = 0.25; shin = 40.; roof = true; }
      else if (p.y > tw.w + 0.3 && ((rf > 0.5 && rf < 1.5) || rf > 4.5)) roof = true;
    } else if ((rt == 1. || rt == 3.) && n.y > 0.3) roof = true;
    else if (rt == 0. && n.y > 0.7) alb *= 0.7;
    if (roof && spec == 0.) { alb = vec3(0.045, 0.05, 0.06) * (0.85 + 0.3 * vnoise(p.xz * 0.6)); spec = 0.12; shin = 24.; }
    if (!roof && abs(n.y) < 0.3) {
      float lit; float wm = windowMask(p, wx, 192., wtop, vec2(3.2, 4.1), pix, lit);
      alb = mix(alb, vec3(0.012), wm * 0.85);
      spec = max(spec, wm * 0.6); shin = max(shin, wm * 90.);
      emi += wm * lit * uDusk * vec3(1., 0.6, 0.25) * 0.9;
    }
    ao = 0.65 + 0.35 * smoothstep(188., 200., p.y);
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
      emi += wm * lit * uDusk * vec3(1., 0.62, 0.28) * 0.7;
    }
  } else if (r.mat == 6) {                            // boundary wall / hedges
    if (r.aux > 0.5) alb = vec3(0.13, 0.125, 0.115) * (0.8 + 0.4 * vnoise(p.xz * 2. + p.y));
    else { alb = vec3(0.03, 0.058, 0.02); n = normalize(n + (vec3(vnoise(p.xz * 3.), vnoise(p.zy * 3.), vnoise(p.xy * 3.)) - 0.5) * 0.8); wrap = 0.3; }
  } else {                                            // set-pieces
    float sid = r.aux;
    if (sid < 1.5) { alb = vec3(0.55, 0.38, 0.09); spec = 0.8; shin = 60.; }
    else if (sid < 3.5) {
      alb = vec3(0.33, 0.018, 0.012); spec = 0.35; shin = 50.;
      if (sid < 2.5 && n.y > 0.5) alb = vec3(0.04);
      if (sid > 2.5 && p.y > TRAIN_Y + 3.55) alb = vec3(0.02);
      if (sid < 2.5 && abs(n.y) < 0.4) {
        float lit; float wm = windowMask(p, faceCoord(p, n, false), TRAIN_Y + 2.2, TRAIN_Y + 3.4, vec2(1.6, 1.2), pix, lit);
        alb = mix(alb, vec3(0.03), wm); emi += wm * uDusk * vec3(1., 0.7, 0.35) * 0.8;
      }
    }
    else if (sid < 4.5) alb = vec3(0.045, 0.035, 0.025);
    else if (sid < 5.5) alb = vec3(0.03, 0.025, 0.02);
    else if (sid < 6.5) alb = vec3(0.21, 0.2, 0.185);
    else if (sid < 7.5) { alb = vec3(0.62, 0.61, 0.58); spec = 0.3; shin = 80.; }
    else {
      vec3 q = p - vec3(WIL.x, WIL_Y, WIL.y);
      alb = (q.y < 4.5 && length(q.xz) < 2.2) ? vec3(0.06, 0.045, 0.03) : vec3(0.045, 0.07, 0.022);
      wrap = 0.25;
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

void main() {
  vec2 uv = (2. * (gl_FragCoord.xy + uJitter) - uRes) / uRes.y;
  vec3 rd = normalize(uCamRot * vec3(uv * uTanHalf, 1.));
  vec3 ro = uCamPos;
  Res r = trace(ro, rd);
  if ((uDbg & 16) != 0) { outColor = vec4(vec3(r.it / 400.), -1.); return; }
  if ((uDbg & 32) != 0) { outColor = vec4(vec3(r.t / 8000.), -1.); return; }
  vec3 col = shade(ro, rd, r);
  if (r.mat != 0) col = applyFog(col, ro, rd, r.t);
  col = pow(aces(col * uExposure), vec3(1. / 2.2));
  outColor = vec4(col, r.mat == 0 ? -1. : r.t);      // alpha: hit distance for reprojection (-1 = sky)
}
