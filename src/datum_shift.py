"""The registered world raised by a datum shift: every absolute height moves by the same amount, exactly in each layer's
quantum. A builder or a test that measures from a fixed level, rather than from the lake or the castle platform, fails
on a shifted world, so the scene can move to a site at another height. The asset tests' shifted runs read the world
through this (src/assets_testing.py), as do the parts that stage S1 builds for them (tools/shifted_part.py)."""
import json

from mesh_source import decode, encode_height

ABSOLUTE_LAYERS = ("locBed", "locCB", "MBed", "ABed", "BBed")   # heights above sea level; the water layers are depths


def shifted(data, dy):
    """A copy of the scene data with every absolute height raised by dy metres, exactly in each layer's quantum."""
    out = {"tex": dict(data["tex"]), "meta": json.loads(json.dumps(data["meta"]))}
    for name in ABSOLUTE_LAYERS:
        layer = data["tex"][name]
        steps = dy / layer["q"]
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError(f"a shift of {dy} m is not a whole number of {name}'s {layer['q']} m steps")
        out["tex"][name] = encode_height(decode(layer) + dy, layer["q"])
    meta = out["meta"]
    meta["lake"] += dy
    meta["pitch"]["level"] += dy
    meta["train"]["level"] += dy
    for tower in meta["towers"]:
        tower[3] += dy
    return out
