"""The winged boar statue that stands on each pillar of the front gates (grounds family, mesh_grounds.gates).

    from tools.blender import fetch
    fetch.run("tools/blender/grounds_boar.py", "src/assets/grounds/boar.npz")

Runs in Blender's own Python. A carved stone boar standing four-square, its head a little lowered, with tusks, a crest
along the spine and a pair of feathered wings raised from the shoulders, modelled from metaballs (the body) and layered
feather blades (the wings), decimated to a few thousand triangles. Everything here is fixed (no randomness), and the
export rounds coordinates and writes the archive with fixed timestamps, so a rerun reproduces it byte for byte.

The export is in the statue's local frame, in metres: x forward (out of the snout), y up from the soles of the hooves,
z to the boar's left, a proper rotation of Blender's (x forward, y left, z up) frame. Arrays: pos (n, 3) float32, nrm
(n, 3) float32 unit vertex normals, idx (t, 3) uint32 triangles wound counter-clockwise about their normals.
"""
import io
import math
import sys
import zipfile

import bmesh
import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

BODY_TRIANGLES = 3200
RESOLUTION = 0.022
EPOCH = (1980, 1, 1, 0, 0, 0)


def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def body():
    """The boar's body as one metaball: torso, shoulder hump, haunches, head and snout, ears, legs and a crest."""
    mb = bpy.data.metaballs.new("boar")
    mb.resolution = RESOLUTION
    mb.render_resolution = RESOLUTION
    mb.threshold = 0.6

    def el(kind, co, radius, size=(1.0, 1.0, 1.0), rot=(0.0, 0.0, 0.0), stiffness=2.0):
        e = mb.elements.new(type=kind)
        e.co, e.radius, e.stiffness = Vector(co), radius, stiffness
        e.size_x, e.size_y, e.size_z = size
        e.rotation = Euler(rot).to_quaternion()
        return e

    el("ELLIPSOID", (0.0, 0.0, 0.68), 0.55, (1.3, 0.7, 0.75))                # barrel of the body
    el("BALL", (0.3, 0.0, 0.82), 0.4)                                          # the high shoulders
    el("BALL", (-0.42, 0.0, 0.72), 0.38)                                       # haunches
    el("ELLIPSOID", (0.76, 0.0, 0.66), 0.3, (1.2, 0.75, 0.85), (0.0, 0.35, 0.0))      # head, lowered
    el("CAPSULE", (1.0, 0.0, 0.52), 0.1, (0.15, 1.0, 1.0), (0.0, 0.55, 0.0), 2.5)     # snout
    for s in (1.0, -1.0):
        el("ELLIPSOID", (0.66, s * 0.12, 0.83), 0.15, (0.5, 0.4, 1.0), (s * 0.4, -0.3, 0.0), 2.5)   # ears
        el("CAPSULE", (0.36, s * 0.17, 0.4), 0.1, (0.3, 1.0, 1.0), (0.0, math.pi / 2, 0.0))          # forelegs
        el("CAPSULE", (-0.46, s * 0.18, 0.41), 0.11, (0.3, 1.0, 1.0), (0.0, math.pi / 2 - 0.12, 0.0))  # hind legs
        el("BALL", (0.37, s * 0.17, 0.09), 0.1, stiffness=2.0)                                       # hooves
        el("BALL", (-0.5, s * 0.18, 0.09), 0.105, stiffness=2.0)
    for k in range(7):                                                        # the bristled crest along the spine
        x = 0.5 - k * 0.15
        el("ELLIPSOID", (x, 0.0, 1.06 - 0.012 * k * k), 0.1, (1.4, 0.45, 1.0), stiffness=1.6)
    el("CAPSULE", (-0.64, 0.0, 0.8), 0.065, (0.12, 1.0, 1.0), (0.0, -0.9, 0.0), 2.5)               # the tail
    obj = link(bpy.data.objects.new("boar", mb))
    return obj


def tusks():
    """Two upcurved tusks either side of the snout, each a tapering cone bent in three segments."""
    parts = []
    for s in (1.0, -1.0):
        bm = bmesh.new()
        ring_prev, rings = None, []
        path = [(1.02, s * 0.1, 0.45), (1.09, s * 0.13, 0.49), (1.12, s * 0.15, 0.55), (1.1, s * 0.14, 0.62)]
        radii = (0.026, 0.021, 0.013, 0.0)
        for i, (p, r) in enumerate(zip(path, radii)):
            d = Vector(path[min(i + 1, 3)]) - Vector(path[max(i - 1, 0)])
            d.normalize()
            u = d.cross(Vector((0.0, 1.0, 0.0))).normalized()
            v = d.cross(u).normalized()
            ring = [bm.verts.new(Vector(p) + r * (math.cos(a) * u + math.sin(a) * v))
                    for a in (2 * math.pi * k / 8 for k in range(8))] if r > 0 else [bm.verts.new(Vector(p))]
            rings.append(ring)
        for a, b in zip(rings[:-1], rings[1:]):
            for k in range(8):
                if len(b) == 1:
                    bm.faces.new((a[k], a[(k + 1) % 8], b[0]))
                else:
                    bm.faces.new((a[k], a[(k + 1) % 8], b[(k + 1) % 8], b[k]))
        bm.faces.new(list(reversed(rings[0])))
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        me = bpy.data.meshes.new("tusk")
        bm.to_mesh(me)
        bm.free()
        parts.append(link(bpy.data.objects.new("tusk", me)))
    return parts


def wing_sheet(bm, lead, trail, camber, normal):
    """A cambered sheet of quads between matching points along a leading and a trailing edge, two rows deep."""
    rows = []
    for a, b in zip(lead, trail):
        mid = a.lerp(b, 0.45) + normal * camber
        rows.append([bm.verts.new(a), bm.verts.new(mid), bm.verts.new(b)])
    for r0, r1 in zip(rows[:-1], rows[1:]):
        for k in range(2):
            bm.faces.new((r0[k], r1[k], r1[k + 1], r0[k + 1]))


def wing(s):
    """A raised, half-spread wing on the left (s = 1) or right (s = -1) shoulder: a feathered sheet from the bony
    leading edge back to a trailing edge scalloped by the flight feathers (the primaries longest towards the tip), a
    layer of shorter coverts over it, and the arm itself; made solid."""
    bones = [Vector((0.25, s * 0.2, 0.98)), Vector((0.05, s * 0.58, 1.26)), Vector((0.1, s * 0.86, 1.56)),
             Vector((-0.14, s * 1.0, 1.78))]

    def lead(f):                                                               # along the arm, f from 0 to 1
        x = f * 3
        k = min(int(x), 2)
        return bones[k].lerp(bones[k + 1], x - k)

    back = Vector((-0.82, s * 0.05, -0.57)).normalized()                       # feathers sweep back and down
    normal = Vector((0.0, s * 1.0, -0.2)).normalized()
    bm = bmesh.new()
    feathers = 15
    fs = [i / (2 * feathers) for i in range(2 * feathers + 1)]
    depth = lambda f: 0.32 + 0.55 * math.sin(math.pi * min(f / 0.82, 1.0) * 0.5) * (1.0 - 0.6 * max(f - 0.82, 0) / 0.18)
    trail = [lead(f) + back.lerp(Vector((-0.45, s * 0.1, 0.89)).normalized(), f ** 2 * 0.55) * (depth(f) + (0.07 if i % 2 else 0.0))
             for i, f in enumerate(fs)]
    wing_sheet(bm, [lead(f) for f in fs], trail, 0.04, normal)
    coverts = 22                                                               # the coverts, a shorter scalloped layer on top
    cs = [i / (2 * coverts) for i in range(2 * coverts + 1)]
    lead_c = [lead(f) + normal * 0.03 for f in cs]
    trail_c = [a + (b - a) * (0.42 + (0.05 if i % 2 else 0.0)) for i, (a, b) in
               enumerate(zip(lead_c, [lead(f) + back * depth(f) + normal * 0.03 for f in cs]))]
    wing_sheet(bm, lead_c, trail_c, 0.02, normal)
    me = bpy.data.meshes.new("wing")
    bm.to_mesh(me)
    bm.free()
    obj = link(bpy.data.objects.new("wing", me))
    solid = obj.modifiers.new("solid", "SOLIDIFY")
    solid.thickness, solid.offset = 0.035, 0.0
    split = obj.modifiers.new("split", "EDGE_SPLIT")
    split.split_angle = math.radians(35.0)
    arm = bmesh.new()                                                          # the wing's bony leading edge
    for a, b in zip(bones[:-1], bones[1:]):
        bmesh.ops.create_cone(arm, cap_ends=True, segments=8, radius1=0.06, radius2=0.045, depth=(b - a).length,
                              matrix=Matrix.Translation((a + b) / 2) @ (b - a).to_track_quat("Z", "Y").to_matrix().to_4x4())
    me = bpy.data.meshes.new("arm")
    arm.to_mesh(me)
    arm.free()
    return [obj, link(bpy.data.objects.new("arm", me))]


def decimated(obj, triangles):
    """The metaball's mesh, decimated to about `triangles`."""
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
    mesh_obj = link(bpy.data.objects.new("body", me))
    bpy.data.objects.remove(obj, do_unlink=True)
    count = sum(len(p.vertices) - 2 for p in me.polygons)
    dec = mesh_obj.modifiers.new("decimate", "DECIMATE")
    dec.ratio = min(1.0, triangles / max(count, 1))
    return mesh_obj


def gather(objects):
    """(pos, nrm, idx) of the objects' evaluated, triangulated meshes in the statue's local frame."""
    dg = bpy.context.evaluated_depsgraph_get()
    to_local = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))   # Blender (x, y, z) -> (x, z, -y)
    pos, nrm, idx, base = [], [], [], 0
    for obj in objects:
        me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.transform(to_local @ obj.matrix_world)
        bm.normal_update()
        bm.verts.index_update()
        pos.append(np.array([v.co[:] for v in bm.verts], np.float64))
        nrm.append(np.array([v.normal[:] for v in bm.verts], np.float64))
        idx.append(np.array([[v.index for v in f.verts] for f in bm.faces], np.int64) + base)
        base += len(bm.verts)
        bm.free()
    pos, nrm, idx = np.concatenate(pos), np.concatenate(nrm), np.concatenate(idx)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    face = np.cross(pos[idx[:, 1]] - pos[idx[:, 0]], pos[idx[:, 2]] - pos[idx[:, 0]])
    flip = np.einsum("ij,ij->i", face, nrm[idx].sum(axis=1)) < 0
    idx[flip] = idx[flip][:, [0, 2, 1]]
    pos[:, 1] -= pos[:, 1].min()                                               # the soles of the hooves at y = 0
    return np.round(pos, 4).astype(np.float32), np.round(nrm, 4).astype(np.float32), idx.astype(np.uint32)


def save(path, **arrays):
    """An .npz archive that is the same bytes on every run: fixed member order and timestamps, no compression."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            z.writestr(zipfile.ZipInfo(f"{name}.npy", date_time=EPOCH), buf.getvalue())


def main(out):
    clear()
    objects = [decimated(body(), BODY_TRIANGLES)] + tusks() + wing(1.0) + wing(-1.0)
    pos, nrm, idx = gather(objects)
    save(out, pos=pos, nrm=nrm, idx=idx)
    print(f"boar: {len(pos)} vertices, {len(idx)} triangles, {np.ptp(pos, axis=0).round(2).tolist()} m -> {out}")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1])
