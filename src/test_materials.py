"""The material registry: one owner per code, codes inside their family's reserved ranges, and shading generated for
every set piece."""
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import build_meshes  # noqa: E402
import materials  # noqa: E402
from materials import Material, set_piece  # noqa: E402
from mesh_source import BUILDING_KINDS  # noqa: E402


def frag_source():
    return "".join((ROOT / "src" / f"frag{i}.glsl").read_text(encoding="utf-8") for i in range(1, 6))


SHADED = "\n        shaded = true; }"


def p3_runs(chain):
    """For each code 0-255, the bodies the generated set-piece chain runs for it, in order: each branch whose condition
    holds, which marks the code shaded when its body ends by saying so, and the default's under !shaded."""
    heads = list(re.finditer(r"^      if \(", chain, re.M))
    branches = []
    for n, head in enumerate(heads):
        depth, i = 1, head.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(chain[i], 0)
            i += 1
        end = heads[n + 1].start() if n + 1 < len(heads) else len(chain)
        branches.append((chain[head.end():i - 1], chain[i:end].rstrip("\n")))
    runs = {}
    for k in range(256):
        shaded, ran = False, []
        for condition, rest in branches:
            if condition == "!shaded":
                if not shaded:
                    ran.append(rest)
            elif k in {int(c) for c in re.findall(r"k == (\d+)", condition)}:
                marks = rest.startswith(" {") and rest.endswith(SHADED)
                ran.append(rest[2:-len(SHADED)] if marks else rest)
                shaded = shaded or marks
        runs[k] = ran
    return runs


def ladder_lines(chain):
    """The lines of the chain that would make it a ladder: an `else` at the chain's own indent. A body's own if and else
    sit deeper, as frag5.glsl's do."""
    return re.findall(r"^      (?:\} *)?else\b.*$", chain, re.M)


def split_top(text):
    """`text` split at the commas outside any parentheses."""
    parts, depth, start = [], 0, 0
    for i, c in enumerate(text):
        depth += {"(": 1, ")": -1}.get(c, 0)
        if c == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    return parts + [text[start:].strip()]


def model_arrays(body):
    """{parameter: (first code, [value per code from the first])} of a model's generated branch."""
    out = {}
    for kind, name, items in re.findall(r"const (\w+) (\w+)\[\d+\] = \w+\[\d+\]\((.*?)\);(?= const| [a-z])", body):
        first = int(re.search(rf"{name}\[k - (\d+)\]", body).group(1))
        out[name.split("_", 1)[1].lower()] = (first, split_top(items))
    return out


def with_material(family, material):
    families = materials.discover()
    ranges, mats = families[family]
    families[family] = (ranges, mats + (material,))
    return families


class RegistryTests(unittest.TestCase):
    def test_the_registry_validates(self):
        mats = materials.materials()
        self.assertTrue(any(m.default for m in mats))
        self.assertEqual(len({m.name for m in mats}), len(mats))

    def test_two_families_claiming_one_code_fail(self):
        families = materials.discover()
        ranges, mats = families["village"]
        families["village"] = (ranges, mats + (set_piece(26, "village-weatherboard", refl="vec3(0.1)"),))
        with self.assertRaisesRegex(ValueError, "reserved by grounds"):
            materials.validate(families)

    def test_one_family_claiming_a_code_twice_fails(self):
        families = materials.discover()
        ranges, mats = families["grounds"]
        families["grounds"] = (ranges, mats + (set_piece(26, "second-weatherboard", refl="vec3(0.1)"),))
        with self.assertRaisesRegex(ValueError, "both claim"):
            materials.validate(families)

    def test_overlapping_reservations_fail(self):
        families = materials.discover()
        families["stub"] = (((7, 55, 65),), ())
        with self.assertRaisesRegex(ValueError, "both reserve"):
            materials.validate(families)

    def test_a_code_outside_every_reservation_fails(self):
        families = materials.discover()
        families["stub"] = ((), (set_piece(250, "stub-paint", refl="vec3(0.1)"),))
        with self.assertRaisesRegex(ValueError, "outside every family's ranges"):
            materials.validate(families)

    def test_shading_text_belongs_to_set_pieces(self):
        families = materials.discover()
        ranges, mats = families["castle"]
        families["castle"] = (ranges, mats + (Material("castle-paint", 4, 46, refl="vec3(0.1)"),))
        with self.assertRaisesRegex(ValueError, "set pieces only"):
            materials.validate(families)

    def test_building_kinds_follow_the_layer_codes(self):
        names = {m.aux: m.name for m in materials.materials() if m.mat == build_meshes.MAT_BUILDING}
        self.assertEqual([names[k] for k in range(len(BUILDING_KINDS))], [f"building-{k}" for k in BUILDING_KINDS])


class ModelTests(unittest.TestCase):
    """Set pieces sharing a surface model share one shading branch, each code reading its own values from it."""
    FLAT = {"alb": "vec3(0.1, 0.2, 0.3)", "spec": "0.5", "shin": "60."}

    def test_a_model_reads_each_code_its_own_values(self):
        families = materials.discover()
        ranges, mats = families["transport"]
        paints = {100: "vec3(0.1, 0.2, 0.3)", 102: "vec3(0.4, 0.5, 0.6)"}
        families["transport"] = (ranges, mats + tuple(
            set_piece(aux, f"test-paint-{aux}", model="flat", values={**self.FLAT, "alb": alb}, refl="vec3(0.1)")
            for aux, alb in paints.items()))
        mats = materials.validate(families)
        arrays = model_arrays(materials.model_bodies(mats)["flat"])
        for m in mats:
            if m.model == "flat":
                for param, value in m.values:
                    with self.subTest(material=m.name, param=param):
                        first, items = arrays[param]
                        self.assertEqual(items[m.aux - first], value)
        first, items = arrays["alb"]
        self.assertEqual([items[aux - first] for aux in paints], list(paints.values()))

    def test_every_modelled_set_piece_reads_its_own_values(self):
        mats = materials.materials()
        bodies = materials.model_bodies(mats)
        for m in mats:
            if m.model:
                arrays = model_arrays(bodies[m.model])
                for code in range(m.aux, m.aux + m.count):
                    for param, value in m.values:
                        with self.subTest(material=m.name, code=code, param=param):
                            first, items = arrays[param]
                            self.assertEqual(items[code - first], value)

    def test_a_new_code_of_a_known_model_adds_no_branch(self):
        def branches(*auxes):
            families = materials.discover()
            ranges, mats = families["transport"]
            families["transport"] = (ranges, mats + tuple(
                set_piece(aux, f"test-paint-{aux}", model="flat", values=tuple(self.FLAT.items()), refl="vec3(0.1)")
                for aux in auxes))
            return len(materials._groups(materials.validate(families), "p3"))
        self.assertEqual(branches(100, 101, 102), branches(100))

    def test_a_modelled_set_piece_gives_exactly_its_models_values(self):
        for values in ({"alb": "vec3(0.1)", "spec": "0.5"}, {**self.FLAT, "rise": "0."}):
            with self.subTest(values=sorted(values)), self.assertRaisesRegex(ValueError, "values"):
                materials.validate(with_material("transport", set_piece(100, "test-paint", model="flat",
                                                                        values=tuple(values.items()), refl="vec3(0.1)")))

    def test_a_model_value_is_a_literal_of_the_parameters_type(self):
        # every set piece shares the shading pass, so a constructor GLSL refuses would stop the page on every GPU
        for value in ("0.5", "vnoise(p.xz)", "vec2(0.1, 0.2)", "vec3(0.1, 0.2)", "vec3(0.1, 0.2, 0.3, 0.4)"):
            values = {**self.FLAT, "alb": value}
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "alb"):
                materials.validate(with_material("transport", set_piece(100, "test-paint", model="flat",
                                                                        values=tuple(values.items()), refl="vec3(0.1)")))

    def test_an_unknown_model_fails(self):
        with self.assertRaisesRegex(ValueError, "no surface model"):
            materials.validate(with_material("transport", set_piece(100, "test-paint", model="velvet",
                                                                    values=tuple(self.FLAT.items()), refl="vec3(0.1)")))

    def test_a_set_piece_has_shading_text_or_a_model_not_both(self):
        with self.assertRaisesRegex(ValueError, "both"):
            materials.validate(with_material("transport", set_piece(100, "test-paint", model="flat", p3=" alb = vec3(0.1);",
                                                                    values=tuple(self.FLAT.items()), refl="vec3(0.1)")))

    def test_more_distinct_set_piece_bodies_than_the_cap_fail(self):
        # past it the shading pass may not start on a GPU, and compiles slowly where it does (REBUILD.md)
        spare = materials.MAX_P3_BRANCHES - len(materials._groups(materials.materials(), "p3"))

        def padded(n):
            families = materials.discover()
            ranges, mats = families["transport"]
            families["transport"] = (ranges, mats + tuple(
                set_piece(100 + i, f"test-paint-{i}", p3=f" alb = vec3(0.{i + 1});", refl="vec3(0.1)") for i in range(n)))
            return materials.validate(families)
        self.assertGreaterEqual(spare, 0)
        self.assertEqual(len(materials._groups(padded(spare), "p3")), materials.MAX_P3_BRANCHES)
        with self.assertRaisesRegex(ValueError, "cap"):
            padded(spare + 1)


class GeneratedShadingTests(unittest.TestCase):
    def setUp(self):
        self.mats = materials.materials()

    def test_every_code_runs_exactly_its_own_shading_body_or_the_defaults(self):
        shared = materials.model_bodies(self.mats)
        own = {a: shared[m.model] if m.model else m.p3 for m in self.mats
               if m.mat == materials.SET_PIECE and (m.p3 is not None or m.model) and not m.default
               for a in range(m.aux, m.aux + m.count)}
        default = next(m.p3 for m in self.mats if m.default)
        chain = materials.p3_branches(self.mats)
        self.assertTrue(chain.startswith("      bool shaded = false;\n"))
        for k, bodies in p3_runs(chain).items():
            with self.subTest(k=k):
                self.assertEqual(bodies, [own.get(k, default)])

    def test_the_set_piece_branches_are_independent(self):
        # an else-if ladder compiles ever more slowly with its depth, and past a depth it does not start (REBUILD.md)
        self.assertEqual(ladder_lines(materials.p3_branches(self.mats)), [])

    def test_a_bodys_own_else_is_not_taken_for_a_ladder(self):
        body = " {\n        if (n.y > 0.5) alb = vec3(0.1);\n        else alb = vec3(0.2);\n      }   // its own if and else"
        mats = materials.validate(with_material("transport", set_piece(100, "test-paint", p3=body, refl="vec3(0.1)")))
        chain = materials.p3_branches(mats)
        self.assertIn(body, chain)
        self.assertEqual(ladder_lines(chain), [])

    def test_every_set_piece_albedo_reaches_the_reflections(self):
        lines = materials.refl_lines(self.mats)
        for m in self.mats:
            if m.mat == materials.SET_PIECE and m.refl is not None and not m.default:
                with self.subTest(material=m.name):
                    self.assertRegex(lines, rf"if \(([^)]* \|\| )?k == {m.aux}( \|\| [^)]*)?\) return {re.escape(m.refl)};")

    def test_every_shaded_set_piece_has_its_own_reflection(self):
        # a set piece without one mirrors in the lake as the default does, in the sails' pale canvas
        for m in self.mats:
            if m.mat == materials.SET_PIECE and (m.p3 is not None or m.model) and not m.default:
                with self.subTest(material=m.name):
                    self.assertIsNotNone(m.refl)

    def test_lit_glazing_glows_in_both_passes(self):
        glow = materials.lamp_lines(self.mats)
        for m in self.mats:
            if m.glow:
                with self.subTest(material=m.name):
                    self.assertIn(f"k == {m.aux} ? {m.glow}", glow)

    def test_each_marker_occurs_once_in_the_sources(self):
        self.assertEqual(frag_source().count(materials.P3_MARKER), 1)
        part_b = (ROOT / "src" / "template_b.html").read_text(encoding="utf-8")
        self.assertEqual(part_b.count(materials.REFL_MARKER), 1)
        self.assertEqual(part_b.count(materials.LAMP_MARKER), 1)

    def test_a_missing_marker_is_refused(self):
        with self.assertRaisesRegex(ValueError, "occurs 0 times"):
            materials.fill("no markers here", materials.P3_MARKER)

    def test_every_code_in_the_payload_is_registered(self):
        self.assertEqual(sorted(c for c in payload_codes() if not materials.registered(*c)), [])

    def test_every_registered_material_is_one_a_mesh_can_carry(self):
        # the meshes replaced the raymarched objects in every pass: a code that only they wrote would be dead shading
        meta = json.loads(build_meshes.strip_js((ROOT / "build" / "data.js").read_text(encoding="ascii")))["meta"]
        generic = {(build_meshes.MAT_CASTLE, build_meshes.CASTLE_CLASS_AUX + k) for k in range(len(meta["castle"]["cls"]))}
        generic |= {(build_meshes.MAT_BUILDING, k) for k in range(len(BUILDING_KINDS))}    # the fallback massing's codes
        written = set(next(m for m in self.mats if m.name == "castle-roof-type").codes)   # the mesh shader's, for P3
        carried = payload_codes() | generic | written
        self.assertEqual(sorted(m.name for m in self.mats if not set(m.codes) & carried), [])


def payload_codes():
    """The (material, aux) codes the built payload's vertices carry."""
    payload = build_meshes.decode_payload((ROOT / "build" / "meshes.js").read_text(encoding="utf-8"))
    return {tuple(int(v) for v in a) for c in payload["chunks"] for a in {tuple(x) for x in c["attr"].tolist()}}


if __name__ == "__main__":
    unittest.main()
