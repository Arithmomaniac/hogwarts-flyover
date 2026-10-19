// ------------------------------------------------------------------ castle
vec2 casUV(vec2 xz) { return vec2(dot(uCasU, vec3(xz, 1.)), dot(uCasV, vec3(xz, 1.))); }

float merlon(float ang, float r) { float n = floor(2. * PI * r / 2.2); return step(0.5, fract(ang / (2. * PI) * n)); }

float towerD(vec3 p, int i) {
  vec4 a = uTw[i]; float rf = uTx[i].x;
  vec2 q = p.xz - a.xy; float rad = length(q); float r = a.z, top = a.w;
  float d;
  if (rf > 2.5 && rf < 4.5) {                // Astronomy / North Tower: thin column, wider overhanging top
    float cr = r * 0.62, dr = r * 1.3, cy = top - 8.;
    d = max(rad - cr, p.y - top);
    d = min(d, max(rad - dr, max(p.y - top, cy - p.y)));
    float k = (dr - cr) / 4.;
    d = min(d, 0.7 * max(rad - (cr + max(0., p.y - (cy - 4.)) * k), max(p.y - cy, cy - 4. - p.y)));
    if (rf < 3.5) {
      float par = max(max(rad - dr, dr - 0.7 - rad), max(p.y - (top + 1.1 + 0.9 * merlon(atan(q.y, q.x), dr)), top - 0.5 - p.y));
      d = min(d, par);
    } else {
      vec3 o = vec3(q.x, p.y - (top + dr * 0.85), q.y);
      d = min(d, sdEll(o, vec3(dr * 1.1, dr * 1.25, dr * 1.1)));
      d = min(d, sdCone(vec2(rad, p.y - (top + dr * 1.8)), dr * 0.35, dr * 1.8));
    }
  } else if (rf > 4.5) {                     // Gryffindor Tower: wide base topped by three trefoil turrets
    d = max(rad - r, p.y - top);
    d = min(d, max(max(rad - r, r - 0.7 - rad), max(p.y - (top + 1.1 + 0.9 * merlon(atan(q.y, q.x), r)), top - 0.5 - p.y)));
    for (int k = ZERO; k < 3; k++) {
      float an = float(k) * 2.0944 + 0.5;
      vec2 c = q - vec2(cos(an), sin(an)) * r * 0.5;
      float tr = r * 0.44, tt = top + 15.;
      d = min(d, max(length(c) - tr, max(p.y - tt, top - 2. - p.y)));
      d = min(d, sdCone(vec2(length(c), p.y - tt), tr + 0.4, 2.4 * tr));
    }
  } else {
    d = max(rad - r, p.y - top);
    if (rf > 0.5) d = min(d, sdCone(vec2(rad, p.y - top), r + 0.5, 2.3 * r));
    else d = min(d, max(max(rad - r, r - 0.7 - rad), max(p.y - (top + 1.1 + 0.9 * merlon(atan(q.y, q.x), r)), top - 0.5 - p.y)));
  }
  return d;
}

// one extruded mass class: footprint signed distance s, absolute top y0, roof type rt (m: battlement checker).
// Divisors are Lipschitz bounds for the roof slope plus the base-height slope (<= 0.45).
float massD(vec3 p, float s, float y0, float rt, float m) {
  float y = y0, sl = 1.1;
  if (rt == 1.) { y += clamp(0.3 - s, 0., 5.5) * 0.95; sl = 1.72; }
  else if (rt == 3.) { y += clamp(0.3 - s, 0., 8.5) * 1.35; sl = 2.06; }
  else if (rt == 0.) y += s > -0.8 ? 1.0 + 0.8 * m : 0.;
  return max(s, (p.y - y) / sl);
}

// union of the extruded mass classes (each with its own footprint, so steps between ranges are real walls)
// and the two nearest analytic towers. cb: castle base height. part: 0 = mass, >0 = tower index + 1
float castleD(vec3 p, float cb, out float top, out float rt, out int part) {
  vec2 uv = casUV(p.xz);
  top = 0.; rt = -1.; part = 0;
  vec2 e = max(max(-uv, uv - 1.), 0.) * CAS_M;
  float dBox = max(length(e), p.y - CAS_TOP);
  if (dBox > 0.) return dBox + 1.;                    // content sits >= 1 m inside the box; keeps bound > hit epsilon
  const vec2 CLS[6] = CAS_CLS;                        // per class: top above the base, roof type
  vec4 sa = textureLod(tCasA, uv, 0.) * 25.5 - 12.9;   // q8 sd, grown 10 cm so neighbouring classes overlap
  vec2 sb = textureLod(tCasB, uv, 0.).rg * 25.5 - 12.9;
  float m = float((fract(p.x * 0.4) < 0.5) != (fract(p.z * 0.4) < 0.5));
  float d = 1e5, dk;
#define CAS_K(k, s) dk = massD(p, s, cb + CLS[k].x, CLS[k].y, m); if (dk < d) { d = dk; top = cb + CLS[k].x; rt = CLS[k].y; }
  CAS_K(0, sa.x) CAS_K(1, sa.y) CAS_K(2, sa.z) CAS_K(3, sa.w) CAS_K(4, sb.x) CAS_K(5, sb.y)
#undef CAS_K
  ivec2 ts = textureSize(tCasB, 0);
  vec2 ti = texelFetch(tCasB, clamp(ivec2(uv * vec2(ts)), ivec2(0), ts - 1), 0).ba * 255. + 0.5;
  for (int k = ZERO; k < 2; k++) {
    int i = int(k == 0 ? ti.x : ti.y);
    float dt = towerD(p, i);
    if (dt < d) { d = dt; part = i + 1; }
  }
  return d;
}

// ------------------------------------------------------------------ Hogsmeade buildings, Hagrid's hut, glasshouses, stands
vec2 pitchUV(vec2 xz) { vec2 q = xz - PITCH_C; return vec2(dot(q, PITCH_DIR), dot(q, vec2(-PITCH_DIR.y, PITCH_DIR.x))); }

float bldD(vec3 p, GS g, float gy, out float kind) {
  kind = -1.;
  float sd = bldSd(g);
  if (sd > 4.) return sd - 0.5;
  float code = floor(textureLod(tLocN, g.luv, 0.).g * 255. + 0.5);
  kind = floor(code / 16.);
  float ey = gy + 2.5 + mod(code, 16.) * 0.5;
  if (kind == 5.) {                          // Quidditch stands: tiers rising away from the pitch
    float ro = length(pitchUV(p.xz) / vec2(101.6, 54.6));
    return max(sd, (p.y - (gy + 1.2 + 8.6 * smoothstep(0.83, 0.99, ro))) * 0.6);
  }
  float pitch = kind == 0. ? 1.15 : (kind == 2. ? 0.55 : 0.85);
  float ridge = kind == 2. ? 2.2 : 4.5;
  float roofY = ey + clamp(0.35 - sd, 0., ridge) * pitch;
  float walls = max(sd + 0.3, p.y - ey);
  float roof = max(sd - 0.12, max((p.y - roofY) / sqrt(1. + pitch * pitch), ey - 0.35 - p.y));
  return min(walls, roof);
}

// boundary wall (drawn wider than scale on the essay map -> thinned) and hedges
float wallD(vec3 p, GS g, float gy, out float which) {
  float hs = g.la.z * 25.5 - 12.8;
  float ws = textureLod(tLocB, g.luv, 0.).r * 25.5 - 12.8;
  float dh = max(hs - 0.45, p.y - (gy + 1.5 - 0.6 * max(hs + 0.45, 0.)));
  float dw = max(ws + WALL_IN, p.y - (gy + 2.2));
  which = dw < dh ? 1. : 0.;
  return min(dh, dw);
}

// ------------------------------------------------------------------ canon set-pieces placed from the essay maps
float goalD(vec3 p) {
  vec2 u = pitchUV(p.xz); float y = p.y - PITCH_LEVEL;
  u.x = abs(u.x) - 61.;
  float d = 1e5;
  for (int k = ZERO - 1; k <= 1; k++) {
    vec2 c = u - vec2(0., float(k) * 5.2);
    float hh = 15.2 - abs(float(k)) * 1.6;
    d = min(d, max(length(c) - 0.16, max(y - hh, -y)));
    vec3 h = vec3(c.x, y - hh - 0.8, c.y);
    d = min(d, length(vec2(length(h.yz) - 0.78, h.x)) - 0.09);
  }
  return d;
}

float trainD(vec3 p, out float part) {
  part = 0.;
  vec2 ab = TB - TA; float L = length(ab); vec2 dir = ab / L, nr = vec2(-dir.y, dir.x);
  vec2 q = p.xz - TA; float s = dot(q, dir), c = dot(q, nr); float y = p.y - TRAIN_Y;
  float cell = 19.9; float k = clamp(floor(s / cell), 0., floor(L / cell) - 1.);
  float d = sdRBox(vec3(s - (k + 0.5) * cell, y - 2.3, c), vec3(9.35, 1.8, 1.42), 0.45);
  vec2 eb = EB - EA; float le = length(eb); vec2 ed = eb / le, en = vec2(-ed.y, ed.x);
  vec2 qe = p.xz - EA; float se = dot(qe, ed), ce = dot(qe, en);
  float boiler = max(length(vec2(y - 2.55, ce)) - 1.05, abs(se - le * 0.42) - le * 0.36);
  float cab = sdRBox(vec3(se - le * 0.86, y - 2.4, ce), vec3(le * 0.13, 1.9, 1.4), 0.2);
  float chim = max(length(vec2(se - le * 0.12, ce)) - 0.32, max(y - 4.5, 2.5 - y));
  float frame = sdBox(vec3(se - le * 0.5, y - 0.95, ce), vec3(le * 0.5, 0.55, 1.3));
  float de = min(min(boiler, cab), min(chim, frame));
  if (de < d) { d = de; part = 1.; }
  return d;
}

float shipD(vec3 p, out float part) {
  part = 0.;
  vec2 ab = SB - SA; float L = length(ab); vec2 dir = ab / L, nr = vec2(-dir.y, dir.x);
  vec2 q = p.xz - 0.5 * (SA + SB); float s = dot(q, dir), c = dot(q, nr); float y = p.y - LAKE;
  float hl = L * 0.5;
  float taper = 1. - pow(clamp(abs(s) / hl, 0., 1.), 3.);
  float w = SHIP_W * 0.36 * sqrt(max(taper, 0.)) * (0.7 + 0.3 * smoothstep(-4., 4., y));
  float d = 0.6 * max(abs(c) - w, max(abs(s) - hl, max(y - 4.2 - 2.5 * pow(abs(s) / hl, 2.), -5. - y)));
  float m = 1e5;
  for (int k = ZERO - 1; k <= 1; k++) {
    float ms = float(k) * hl * 0.55;
    m = min(m, max(length(vec2(s - ms, c)) - 0.45, max(y - 54., -y)));
    for (int j = ZERO; j < 3; j++) {
      float yy = 18. + float(j) * 11.;
      m = min(m, sdBox(vec3(s - ms, y - yy, c), vec3(0.6, 0.6, 9. - float(j) * 2.2)));
    }
  }
  if (m < d) { d = m; part = 1.; }
  return d;
}

float gatesD(vec3 p) {
  float d = 1e5;
  for (int k = ZERO; k < 2; k++) {
    vec2 c = k == 0 ? GATE_A : GATE_B;
    vec3 q = vec3(p.x - c.x, p.y - GATE_Y, p.z - c.y);
    d = min(d, sdBox(q - vec3(0., 2.6, 0.), vec3(0.7, 2.6, 0.7)));
    vec3 b = q - vec3(0., 6.0, 0.);
    d = min(d, sdEll(b, vec3(0.95, 0.55, 0.5)));
    d = min(d, sdEll(b - vec3(0.95, 0.15, 0.), vec3(0.35, 0.28, 0.25)));
    vec3 w1 = vec3(b.x, b.y - 0.55, abs(b.z) - 0.55);
    float ca = 0.8, sa = 0.6;
    w1 = vec3(w1.x, ca * w1.y - sa * w1.z, sa * w1.y + ca * w1.z);
    d = min(d, sdEll(w1, vec3(0.7, 0.75, 0.07)));
  }
  return d;
}

float willowD(vec3 p) {
  vec3 q = vec3(p.x - WIL.x, p.y - WIL_Y, p.z - WIL.y);
  float trunk = max(length(q.xz) - 1.5 + 0.08 * q.y, max(q.y - 7.5, -q.y - 1.));
  float crown = sdEll(q - vec3(0., 8.5, 0.), vec3(9.5, 5.8, 9.5)) + (vnoise(q.xz * 0.55 + q.y * 0.3) - 0.5) * 1.6;
  return min(trunk, crown * 0.7);
}

// sid: 1 goalposts, 2 carriages, 3 engine, 4 ship hull, 5 ship rig, 6 gate pillars + boars, 7 tomb, 9 willow
float specialD(vec3 p, out float sid) {
  float d = 1e5; sid = 0.;
  if (length(p.xz - PITCH_C) < 90.) { float g = goalD(p); if (g < d) { d = g; sid = 1.; } }
  vec2 tm = 0.5 * (TA + EB);
  if (length(p.xz - tm) < 170.) { float pt; float t = trainD(p, pt); if (t < d) { d = t; sid = 2. + pt; } }
  if (length(p.xz - 0.5 * (SA + SB)) < 70.) { float pt; float s = shipD(p, pt); if (s < d) { d = s; sid = 4. + pt; } }
  vec2 gm = 0.5 * (GATE_A + GATE_B);
  if (length(p.xz - gm) < 22.) { float g = gatesD(p); if (g < d) { d = g; sid = 6.; } }
  if (length(p.xz - TOMB) < 12.) { float b = sdRBox(vec3(p.x - TOMB.x, p.y - TOMB_Y - 0.75, p.z - TOMB.y), vec3(1.7, 0.75, 0.85), 0.08); if (b < d) { d = b; sid = 7.; } }
  if (length(p.xz - WIL) < 18.) { float w = willowD(p); if (w < d) { d = w; sid = 9.; } }
  return d;
}

// all hard-surface objects; mat 4 castle, 5 building, 6 wall/hedge, 7 special. full = false: castle only (lake reflections)
float objD(vec3 p, GS g, float gy, bool full, out int mat, out float aux, out float ctop) {
  mat = 4; aux = 0.; ctop = 0.;
  if (full) {
    float far = textureLod(tLocB, g.luv, 0.).b * 255.;   // horizontal distance to the nearest object (m)
    if (far > 14.) return far - 3.;
  }
  float rt; int part;
  float d = castleD(p, g.cb, ctop, rt, part);
  aux = part > 0 ? 100. + float(part) : rt;
  if (!full) return d;
  if (g.lw > 0.01) {
    if (p.y - gy < 26.) {
      float kind; float db = bldD(p, g, gy, kind);
      if (db < d) { d = db; mat = 5; aux = kind; }
      float wh; float dw = wallD(p, g, gy, wh);
      if (dw < d) { d = dw; mat = 6; aux = wh; }
    } else d = min(d, p.y - gy - 19.);
  }
  float sid; float ds = specialD(p, sid);
  if (ds < d) { d = ds; mat = 7; aux = sid; }
  return d;
}
