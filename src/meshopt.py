"""meshoptimizer 1.3's vertex and index-sequence codecs and its simplifier, run from the vendored npm files' own
WebAssembly under wasmtime.

src/vendor/meshoptimizer holds meshopt_decoder.cjs, meshopt_encoder.js, meshopt_simplifier.js and LICENSE.md byte for
byte as the meshoptimizer 1.3.0 npm package ships them (MIT); REBUILD.md says how to refresh them. The page inlines
meshopt_decoder.cjs; the build encodes with meshopt_encoder.js's module and decodes with the page's own decoder module,
so both sides run the same code. The simplifier makes the coarser levels of detail of the families that have none of
their own (mesh_levels.py), at build time only. The pinned hashes below also put the vendored files into the stages'
cache keys.
"""
import atexit
import functools
import hashlib
import re
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "vendor" / "meshoptimizer"
PINNED = {"meshopt_decoder.cjs": "fabafaaa29cbb1cfa367f20e896d7f6e95b86e575e2ea86ddb6f8cca1a87152d",
          "meshopt_encoder.js": "22c128be3e22b187e2d2661882fc62f51e995f409c0fe14c9f6358cfbaff1611",
          "meshopt_simplifier.js": "c3098ee4cb4fc242b84f064c88898e76e44bf58b0e1f693507f323c08cdbb5ed",
          "LICENSE.md": "f03037ca7bad1e3eb7f4a63fa6084a8baabd5ba30d3c239a9a7f35705d873e26"}
CODEC_VERSION = 1                   # the bitstream version both encoders write (1.3's default; its decoder reads 0 and 1)
VERTEX_LEVEL = 2                    # compression level 3 gains nothing on these streams
SIMPLIFY = {"LockBorder": 1, "Sparse": 2, "ErrorAbsolute": 4, "Prune": 8, "Regularize": 16, "Permissive": 32}


def vendored(name):
    """A vendored file's text, checked against its pinned SHA-256."""
    data = (VENDOR / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != PINNED[name]:
        raise ValueError(f"src/vendor/meshoptimizer/{name} does not match its pinned SHA-256; re-vendor it as "
                         "REBUILD.md says and update meshopt.PINNED")
    return data.decode("utf-8")


def wasm(name, var):
    """The WebAssembly module a vendored file embeds in `var`, unpacked as the file's own unpack() does."""
    text = vendored(name)
    blob = re.search(rf"var {var} =\s*'([^']*)'", text).group(1)
    table = [int(v) for v in re.search(r"var wasmpack = new Uint8Array\(\[([^\]]*)\]", text).group(1).split(",")
             if v.strip()]
    codes = [c - 97 if c > 96 else c - 39 if c > 64 else c + 4 for c in blob.encode("ascii")]
    out, i = bytearray(), 0
    while i < len(codes):
        if codes[i] < 60:
            out.append(table[codes[i]])
            i += 1
        else:                       # two codes carry one byte; the JS stores it in a Uint8Array, hence the mask
            out.append(((codes[i] - 60) * 64 + codes[i + 1]) & 0xFF)
            i += 2
    return bytes(out)


class Module:
    """One instance of a vendored module, with its sbrk heap used as the JS wrappers use it."""

    def __init__(self, binary):
        import wasmtime  # noqa: PLC0415 (only the build and the tests need it; assemble.py stays standard library)
        self.store = wasmtime.Store()
        self.exports = wasmtime.Instance(self.store, wasmtime.Module(self.store.engine, binary), []).exports(self.store)
        self.memory = self.exports["memory"]
        self.call("__wasm_call_ctors")

    def call(self, name, *args):
        return self.exports[name](self.store, *args)

    def run(self, name, out_size, source, args):
        """`name` called with (output, *args) where each None in `args` stands for the copied-in `source`; the first
        `out_size` bytes of the output and the call's result."""
        top = self.call("sbrk", 0)
        try:
            out = self.call("sbrk", out_size)
            at = self.call("sbrk", len(source))
            self.memory.write(self.store, source, at)
            result = self.call(name, out, *(at if a is None else a for a in args))
            return bytes(self.memory.read(self.store, out, out + out_size)), result
        finally:
            self.call("sbrk", top - self.call("sbrk", 0))


@functools.cache
def encoder():
    m = Module(wasm("meshopt_encoder.js", "wasm"))
    m.call("meshopt_encodeVertexVersion", CODEC_VERSION)
    m.call("meshopt_encodeIndexVersion", CODEC_VERSION)
    return m


@functools.cache
def decoder():
    return Module(wasm("meshopt_decoder.cjs", "wasm_base"))


@functools.cache
def simplifier():
    return Module(wasm("meshopt_simplifier.js", "wasm"))


@atexit.register
def _release():
    """Free the instances while wasmtime can still free them; at interpreter teardown its bindings are gone."""
    encoder.cache_clear()
    decoder.cache_clear()
    simplifier.cache_clear()


def encode_vertices(data, stride):
    """`data`, whole vertices of `stride` bytes (a multiple of 4), through the vertex codec."""
    if stride <= 0 or stride > 256 or stride % 4 or len(data) % stride:
        raise ValueError(f"{len(data)} bytes are not whole vertices of a {stride}-byte stride (a multiple of 4)")
    count, m = len(data) // stride, encoder()
    bound = m.call("meshopt_encodeVertexBufferBound", count, stride)
    out, size = m.run("meshopt_encodeVertexBufferLevel", bound, bytes(data),
                      (bound, None, count, stride, VERTEX_LEVEL, CODEC_VERSION))
    if not size:
        raise ValueError("the vertex encoder overran its bound")
    return out[:size]


def decode_vertices(data, count, stride):
    """`count` vertices of `stride` bytes from vertex-codec bytes; malformed input raises. A wrong count or stride can
    decode without error (zero and bit-packed lanes cost the same for nearby counts), so callers must pin the count
    some other way."""
    out, res = decoder().run("meshopt_decodeVertexBuffer", ((count + 3) & ~3) * stride, bytes(data),
                             (count, stride, None, len(data)))
    if res:
        raise ValueError(f"malformed vertex stream ({res}) for {count} vertices of {stride} bytes")
    return out[:count * stride]


def encode_indices(indices):
    """16-bit indices, in order, through the index-sequence codec (exact: unlike the triangle codec it never rotates
    a triangle, so the provoking vertex stays)."""
    import numpy as np  # noqa: PLC0415 (assemble.py reads the vendored decoder through this module with the standard library alone)
    indices = np.asarray(indices)
    if indices.size and (indices.min() < 0 or indices.max() > 0xFFFF):
        raise ValueError("indices must fit 16 bits")
    flat, m = indices.astype("<u4").ravel(), encoder()
    bound = m.call("meshopt_encodeIndexSequenceBound", flat.size, int(flat.max()) + 1 if flat.size else 1)
    out, size = m.run("meshopt_encodeIndexSequence", bound, flat.tobytes(), (bound, None, flat.size))
    if not size:
        raise ValueError("the index encoder overran its bound")
    return out[:size]


def decode_indices(data, count):
    """`count` little-endian 16-bit indices from index-sequence bytes; malformed or mismatched input raises."""
    out, res = decoder().run("meshopt_decodeIndexSequence", ((count + 3) & ~3) * 2, bytes(data),
                             (count, 2, None, len(data)))
    if res:
        raise ValueError(f"malformed index stream ({res}) for {count} indices")
    return out[:count * 2]


def simplify(triangles, positions, target_count, target_error, flags=()):
    """meshopt_simplify: `triangles` ((n, 3) indices into `positions`, (v, 3)) reduced towards `target_count` triangles
    while the error stays within `target_error`; the kept triangles (indices into the same positions) and the error
    reached, in metres with flag ErrorAbsolute, otherwise relative to the mesh's extent. `flags` name SIMPLIFY's."""
    import numpy as np  # noqa: PLC0415 (as encode_indices)
    idx = np.ascontiguousarray(triangles, "<u4").ravel()
    pos = np.ascontiguousarray(positions, "<f4").reshape(-1, 3)
    if idx.size and idx.max() >= len(pos):
        raise ValueError("a triangle indexes past its positions")
    m = simplifier()
    top = m.call("sbrk", 0)
    try:
        err, out, src_pos, src_idx = (m.call("sbrk", n) for n in (4, idx.nbytes, pos.nbytes, idx.nbytes))
        m.memory.write(m.store, pos.tobytes(), src_pos)
        m.memory.write(m.store, idx.tobytes(), src_idx)
        count = m.call("meshopt_simplify", out, src_idx, idx.size, src_pos, len(pos), 12, 3 * int(target_count),
                       float(target_error), sum(SIMPLIFY[f] for f in flags), err)
        kept = np.frombuffer(bytes(m.memory.read(m.store, out, out + 4 * count)), "<u4").reshape(-1, 3).copy()
        return kept, float(np.frombuffer(bytes(m.memory.read(m.store, err, err + 4)), "<f4")[0])
    finally:
        m.call("sbrk", top - m.call("sbrk", 0))
