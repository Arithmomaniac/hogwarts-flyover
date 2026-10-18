"""Extra point features + points of interest, merged into build/final_meta.json."""
from common import *
from build_final import wall_chain_polygon, g2w, h2w, DECOR_G


def main():
    meta = json.load(open(P("build", "final_meta.json")))
    mg = dict(np.load(P("build", "native_g.npz")))
    ex = {}

    # --- winged-boar gates (3g): the gap in the boundary wall nearest the label ---
    chain = wall_chain_polygon(mg["wall"])
    lab = np.array([416.0, 1248.0])
    gaps = []
    for a, b in zip(chain[:-1], chain[1:]):
        mid = (a + b) / 2
        if np.linalg.norm(mid - lab) < 140: gaps.append((np.linalg.norm(b - a), a, b))
    gaps.sort(key=lambda g: -g[0])
    _, a, b = gaps[0]
    pa = np.array(g2w(*a), float); pb = np.array(g2w(*b), float)
    ex["gates"] = [pa.tolist(), pb.tolist()]
    print("gates gap %.1f m" % np.linalg.norm(pb - pa), pa, pb)

    # --- Dumbledore's white tomb (2c) on the promontory ---
    ex["tomb"] = [float(v) for v in g2w(1033, 1763)]

    # --- Durmstrang ship (8d): the grey hull drawn to scale in the lake ---
    wood = mg["wood"].copy(); box = np.zeros_like(wood); box[1090:1250, 980:1100] = True
    ys, xs = np.nonzero(wood & box)
    cx, cy = xs.mean(), ys.mean()
    ev, evc = np.linalg.eigh(np.cov(np.stack([xs - cx, ys - cy]))); d = evc[:, 1]
    pr = (xs - cx) * d[0] + (ys - cy) * d[1]; pw = -(xs - cx) * d[1] + (ys - cy) * d[0]
    s0 = np.array(g2w(cx + d[0] * pr.min(), cy + d[1] * pr.min()), float)
    s1 = np.array(g2w(cx + d[0] * pr.max(), cy + d[1] * pr.max()), float)
    width = (np.percentile(pw, 97) - np.percentile(pw, 3)) * 0.72
    ex["ship"] = {"a": s0.tolist(), "b": s1.tolist(), "w": float(width)}
    print("ship length %.1f m width %.1f m" % (np.linalg.norm(s1 - s0), width))

    # --- Whomping Willow (5a) ---
    ex["willow"] = [float(v) for v in g2w(1060, 492)]

    # --- lake centroid ---
    lake = np.load(P("lake_mask_grounds.npy"))
    ly, lx = np.nonzero(lake)
    ex["lake_c"] = [float(v) for v in g2w(lx.mean(), ly.mean())]

    fo = json.load(open(P("feature_offsets.json")))
    named = meta["named"]
    kh = 1735 / 1041
    tb = h2w(300 * kh, 742 * kh)                                       # Three Broomsticks (2c, Hogsmeade map)
    poi = [
        ("Hogwarts Castle", [0.0, 0.0], "eight storeys plus towers on a mound above a ~120 ft cliff (essay)"),
        ("Entrance Hall & Great Hall", named["EH"], "front doors face WNW toward the Forest; the June sunset reaches the steps"),
        ("Astronomy Tower", named["AT"], "thin column with a wider, open, overhanging top beside the main doors"),
        ("Gryffindor Tower", named["GT"], "wide circular base topped with three trefoil turrets (essay plan)"),
        ("West Tower (Owlery)", named["WT"], "flat-topped with battlements: Buckbeak landed on them"),
        ("North Tower", named["NT"], "thin column, overhanging top under an onion dome"),
        ("The Lake", ex["lake_c"], "~1,000 yd long, round the side of the castle, not behind it"),
        ("Hagrid's hut", [-291.5, -191.9], "about 320 yd from the front door, at the Forest edge"),
        ("Quidditch pitch", [-243.6, -485.9], "500 x 180 ft oval, long axis east-west, golden 50 ft hoops"),
        ("Hogsmeade station", [309.3, 117.4], "170 yd platform; the Express drawn to scale on the essay map"),
        ("Whomping Willow", ex["willow"], "alone on the lawn, nearer the Forest than the castle"),
        ("Greenhouses", fo["greenh"], "low-lying, by the lake, behind and left of the castle"),
        ("Winged-boar gates", ((pa + pb) / 2).tolist(), "south-west of the castle; turn left for Hogsmeade"),
        ("Dumbledore's tomb", ex["tomb"], "white tomb on the lakeside promontory"),
        ("Durmstrang ship", ex["ship"]["a"], "moored in the lake (GoF), masts ~180 ft"),
        ("Hogsmeade", [float(tb[0]), float(tb[1])], "thatched village ~1 mile SSW of the castle"),
        ("Shrieking Shack", [-877.0, -883.5], "on its own little hill on the Hogwarts side of the High Street"),
        ("Forbidden Forest", [float(v) for v in ff_to_world(1150, 2300)], "mainly deciduous, ~4 miles N-S, west and north-west of the castle"),
        ("Aragog's hollow", [float(v) for v in ff_to_world(1560, 2134)], "steep-sided crater ~1⅓ miles into the Forest"),
        ("Grawp's clearing", [float(v) for v in ff_to_world(940, 2186)], "artificial clearing among pines"),
        ("Sirius's cave", [float(v) for v in ff_to_world(60, 1790)], "on the mountain above the stile, ~1,300 yd up a stony path"),
    ]
    # real-world references (OSM, TM frame -> castle frame)
    osm = json.load(open(P("osm.json")))
    real = {}
    for el in osm["elements"]:
        t = el.get("tags", {}); n = t.get("name")
        if n in ("Loch Beannacharain", "Scardroy Lodge", "Creag Coire na Feòla", "Bac an Eich", "Sgùrr a' Ghlas Leathaid", "Sgurr a' Ghlas Leathaid", "Meallan nan Uan", "Maoile Lunndaidh", "Sgùrr na Fearstaig", "Sgurr na Fearstaig"):
            if el["type"] == "node": lat, lon = el["lat"], el["lon"]
            else:
                g = el.get("geometry") or []
                if not g: continue
                lat = float(np.mean([p["lat"] for p in g])); lon = float(np.mean([p["lon"] for p in g]))
            e, nn = to_tm.transform(lon, lat)
            real[n] = [float(e - CASTLE_E), float(nn - CASTLE_N), t.get("ele", "")]
    ex["poi"] = [{"name": a, "p": [round(float(b[0]), 1), round(float(b[1]), 1)], "note": c} for a, b, c in poi]
    ex["real"] = {k: [round(v[0], 1), round(v[1], 1), v[2]] for k, v in real.items()}
    lon, lat = to_ll.transform(CASTLE_E, CASTLE_N)
    ex["castle_ll"] = [float(lat), float(lon)]
    print("castle lat/lon", lat, lon)
    print("real refs", ex["real"])
    meta["extras"] = ex
    json.dump(meta, open(P("build", "final_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
