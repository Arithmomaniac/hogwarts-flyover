// ------------------------------------------------------------------ the pitch's frame (the ground's markings and the stands' colours)
// The raymarched castle, buildings, walls, hedges and set pieces are gone: the exterior meshes stand in every pass.
vec2 pitchUV(vec2 xz) { vec2 q = xz - PITCH_C; return vec2(dot(q, PITCH_DIR), dot(q, vec2(-PITCH_DIR.y, PITCH_DIR.x))); }
