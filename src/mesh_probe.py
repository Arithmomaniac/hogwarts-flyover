"""Paired frame-time probe for the mesh upgrade: explicit renders, each batch timed until the GPU has finished it.

The probe instruments a temporary copy of each page (never the shipped file). Once a page is ready its animation loop
stops and the probe renders every frame itself, two in flight, timing each batch until a WebGL fence reports the GPU
done; an idle loop's frame counter cannot stand in for that work. Within a window the configurations run back to back
in a balanced order that changes every round, so each paired ratio compares neighbouring work on the same GPU state.
Every block starts a fresh history and reads its settings back from the shaders' uniforms.

On the AMD iGPU a second live WebGL context runs slower than the first, whatever its page, in the same browser or in
another: 11-34% by window in A/A tests in November 2026, where on NVIDIA the later copy ran 0-1.4% faster. So does a
page opened in a browser that held another page before, by navigation or in a new tab (11-23%); a page opened in a
fresh browser ran at the level of the first. By default (pages, run_pages; LAYOUTS) both pages stay open in each
browser session, and the rounds are split between sessions that alternate the two load orders. The fresh layouts
(run_fresh), on trial for AMD, run one browser at a time, a fresh one for every visit to a page: each round visits the
two pages in its session's load order, and A/A sessions visit the candidate twice, their second visits over their first
(absolute_check) a check for a gross error in fresh visits: a run whose A/A lies beyond TOLERANCE throughout its
interval does not count. A visit times every window (fresh) or one
(fresh-window); timing every window put a window's two pages a minute apart, over which AMD's level drifts. Each ratio's estimate is the geometric mean of the
sessions' medians, which cancels an order effect only when both pages share it. A window's verdict does not rest on
that: the candidate passes the plan's budget (BUDGET) only if it meets it in both load orders, and fails only if it
misses it in both (summarize). Each ratio also gets a bootstrap interval, with checks that the run can show the
variation between browser launches and that the control comes out even. A run exits with status 1 unless every window
passes.

Work the quiet lock cannot see still loads the machine, so the probe measures the cores other processes keep busy:
for a few seconds before it starts its browsers, waiting while they exceed the caps, then over each window in each
session. Security and OS processes (INDUCED: Defender, Global Secure Access, System, svchost) act on the probe's own
browser, scanning what it writes and inspecting its traffic, so their cores count against a separate, looser cap
(--max-induced-cores), and every other process's against --max-other-cores. The caps are absolute. The baseline runs
interleaved with the candidate in every window, so a steady background within them slows both alike, but one beyond
them is no measurement, however long it has been there; a Defender scan or update exceeds the induced cap. A window
over either cap is measured again, up to LOAD_ATTEMPTS times. A run that cannot start, or keeps a window over a cap
through every attempt, exits with status 3, which the stage runner reads as blocked: its ratios are recorded but are
not results. Every attempt's two loads are recorded, and each window's are printed next to its ratios.

The browser leaves the security processes little to scan or inspect: every launch reuses one profile, in the machine's
cache, and writes no disk or GPU shader cache and makes no background traffic.

    python -X utf8 src\\mesh_probe.py --gpu amd --baseline <frozen page> --out probe-amd.json
"""
import argparse
import asyncio
import contextlib
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import struct
import sys
import tempfile
import time

import psutil
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
GPU_ARGS = {"amd": ["--force-low-power-gpu"], "nvidia": ["--force-high-performance-gpu"]}
# what the security processes would otherwise scan or inspect: caches written to disk and the browser's own traffic
QUIET_ARGS = ["--disable-gpu-shader-disk-cache", "--disk-cache-size=1", "--media-cache-size=1",
              "--disable-background-networking", "--disable-component-update", "--disable-sync", "--no-pings",
              "--disable-domain-reliability"]
VIEWPORT = {"width": 1920, "height": 1080}
SETTINGS = {"q": 1, "detail": 1, "scale": 0.5, "tod": 18.5}      # the page's defaults at the default internal scale
PREROLL, COUNT, FPS, PAN = 16, 16, 30, 0.012                   # the research campaign's window shape
IN_FLIGHT = 2
DRAIN_MS = 5                       # a readback after a block that waited for the GPU is a round trip, not a frame
MAX_OTHER_CORES = 2.0              # other processes' busy cores beyond which a measurement is not kept
# security and OS processes' busy cores beyond which a measurement is not kept: in the first quiet window, with the
# probe's browser running, Defender kept about 2.1 cores busy, Global Secure Access 0.9 and System and svchost 0.5
MAX_INDUCED_CORES = 5.0
INDUCED = frozenset(n.lower() for n in (
    "MsMpEng.exe", "MpDefenderCoreService.exe", "NisSrv.exe", "MpDlpService.exe", "SecurityHealthService.exe",
    "MsSense.exe", "SenseNdr.exe", "SenseTracer.exe", "SenseCncProxy.exe", "SenseIR.exe", "SenseSampleUploader.exe",
    "SenseDlpService.exe", "GlobalSecureAccessClient.exe", "GlobalSecureAccessTunnelingService.exe",
    "GlobalSecureAccessEngineService.exe", "GlobalSecureAccessPolicyRetrieverService.exe",
    "System", "Registry", "Memory Compression", "svchost.exe", "SearchIndexer.exe", "SearchProtocolHost.exe",
    "SearchFilterHost.exe"))
LOAD_ATTEMPTS = 3
REFERENCE_S = 6.0                  # each measurement of other processes' load before the browsers start
START_ATTEMPTS = 20                # how many of those the probe waits through for that load to fall within the cap
# tour start times (s) at site X, set before any timing there: one window per mesh family as the flyover frames it,
# plus the research campaign's targeted castle-and-trees windows and its pan, each moved to where X's tour frames the
# same subject. When the tour changes, tools.timing.windows moves them to where the new tour passes the same points.
# The showcase tour passed all but hogsmeade more than 10 m from their places, so each went to the heaviest view of its
# subject in the same stretch of the tour, by the framing test's grid (src/test_render_tour.py): the castle, the ship,
# the station and train, the greenhouses. trees-65 went to the heaviest view of the castle and trees together, and
# trees-60 to the heaviest that keeps it and look-pan, half a second after it as before, at least 2 s from the others.
WINDOWS = {"castle": 58.65, "lake-ship": 51.65, "station-train": 92.4, "greenhouses": 138.6, "hogsmeade": 435.45,
           "trees-60": 60.65, "trees-65": 54.65, "look-pan": 61.15}
PANNED = {"look-pan"}
# name: (page, meshes, debug bits): the frozen baseline; the candidate's landscape alone (meshes off, a diagnostic); the
# candidate as it ships, its meshes the scene's objects; and the same again, an A/A control whose ratio to the first
# shows how far noise alone moves a ratio. Four keep the Williams square balanced (balanced_orders)
CONFIGS = {"baseline": ("baseline", False, 0), "landscape": ("candidate", False, 0), "integrated": ("candidate", True, 0),
           "control": ("candidate", True, 0)}
RATIOS = [("integrated", "baseline"), ("landscape", "baseline"), ("integrated", "landscape"), ("control", "integrated")]
LOAD_ORDERS = (("baseline", "candidate"), ("candidate", "baseline"))    # browser sessions alternate them, rounds split
BUDGET = 1.10                       # the plan's acceptance case: each window's integrated/baseline at most this, per GPU
BAND, DRAWS = 0.90, 2000            # a ratio's bootstrap interval, and its draws
MIN_SESSIONS = 2                    # sessions per load order at least, so an interval holds the variation between launches
# a systematic error the control check lets pass, a fifth of the budget's 10%; the control is flagged only when its
# whole interval lies beyond it, so noise alone seldom flags a window
TOLERANCE = 1.02
ROUNDS, SESSIONS = 16, 4            # the default run: two sessions per load order, each running every configuration order
# the fresh layout's (run_fresh): eight sessions of four rounds, four per load order, and two A/A sessions
FRESH_ROUNDS, FRESH_SESSIONS, AA_SESSIONS = 32, 8, 2
LAYOUTS = {"amd": "pages", "nvidia": "pages"}  # AMD's fresh layouts (run_fresh) are on trial: too noisy so far
LOOP, API = "\n  function loop(now) {\n", "\n  window.__hog = {\n"
ADAPTER = r"""
  // mesh_probe.py's adapter, only ever inserted into a temporary copy of the page
  let probeStop = false, probeRenders = 0;
  const probeRender = render;
  render = function (pos, f) { probeRenders++; return probeRender(pos, f); };
  const probeChannel = new MessageChannel(), probeWaiting = [];
  probeChannel.port1.onmessage = () => probeWaiting.shift()();
  const probeYield = () => new Promise((resolve) => { probeWaiting.push(resolve); probeChannel.port2.postMessage(0); });
  async function probeDone() {                     // a fence's status only changes between tasks: yield, unthrottled
    const fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
    try {
      for (;;) {
        const status = gl.clientWaitSync(fence, 0, 0);
        if (status === gl.WAIT_FAILED) throw new Error("GPU_COMPLETION");
        if (status !== gl.TIMEOUT_EXPIRED) break;
        await probeYield();
      }
    } finally { gl.deleteSync(fence); }
    if (gl.isContextLost() || gl.getError() !== gl.NO_ERROR) throw new Error("WEBGL_ERROR");
  }
  const probeSnapshot = () => ({
    q: gl.getUniform(prog, U.uQual), detail: gl.getUniform(prog, U.uDetail), dbg: gl.getUniform(prog, U.uDbg),
    t: gl.getUniform(prog, U.uTime), scale, tod, internal: [fw, fh], output: [cv.width, cv.height],
    meshes: typeof meshOn === "undefined" ? false : meshOn, drawn: typeof meshDrawn === "undefined" ? 0 : meshDrawn,
    visible: document.visibilityState });
  window.__probe = {
    async stop() { probeStop = true; await probeDone(); },
    settings(s) {
      qual = s.q; detail = s.detail; scale = s.scale; tod = s.tod; dbg = s.dbg;
      if (typeof meshOn === "undefined") { if (s.meshes) throw new Error("NO_MESHES"); }
      else if (meshOn !== s.meshes) { meshOn = s.meshes; meshGen++; }
    },
    async block(frames, inFlight) {                // a fresh history and jitter cycle, then batches timed to completion
      const loop = window.__hog.frames, renders = probeRenders, batches = [];
      histOk = false; jIdx = 0;
      for (let i = 0; i < frames.length; i += inFlight) {
        const start = performance.now();
        for (const { t, pan } of frames.slice(i, i + inFlight)) {
          const p = pathAt(t), c = Math.cos(pan), s = Math.sin(pan);
          simT = t;
          render(p.pos, [p.fwd[0] * c + p.fwd[2] * s, p.fwd[1], p.fwd[2] * c - p.fwd[0] * s]);
        }
        await probeDone();
        batches.push(performance.now() - start);
      }
      const px = new Uint8Array(4), drained = performance.now();   // a readback waits for any work the fences missed
      gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
      return { batches, drainMs: performance.now() - drained, rendered: probeRenders - renders,
        loopFrames: window.__hog.frames - loop, snapshot: probeSnapshot() };
    },
  };
"""


def instrument(source, target):
    """Copy the page with the probe's adapter; refuse a page whose loop and render contract the adapter doesn't know."""
    text = Path(source).read_text(encoding="utf-8")
    if any(text.count(anchor) != 1 for anchor in (LOOP, API, "\n  function render(pos, f) {\n")):
        raise ValueError(f"{Path(source).name} lacks the probe's anchors; the adapter needs updating for this page")
    text = text.replace(LOOP, LOOP + "    if (probeStop) return;\n", 1).replace(API, ADAPTER + API, 1)
    Path(target).write_text(text, encoding="utf-8", newline="\n")


def schedule(start, pan=False):
    """A window's frames: the tour from `start` at 30 fps, preroll then scored, turning 0.012 rad a frame if panned."""
    return [{"t": start + (i - PREROLL) / FPS, "pan": PAN * (i - PREROLL) if pan else 0.0}
            for i in range(PREROLL + COUNT)]


def balanced_orders(n):
    """A Williams square: each configuration takes every position once and directly follows every other once."""
    first = [0] + [(k + 1) // 2 if k % 2 else n - k // 2 for k in range(1, n)]
    return [[(v + r) % n for v in first] for r in range(n)]


def session_order(session, k, n):
    """The row of balanced_orders(n) that a session's k-th round runs. Row r + n/2 is row r reversed, so each two rounds
    run an order and its mirror image, putting every configuration in mirrored positions, and each n rounds run all n
    rows; sessions start one row apart."""
    return (session + k // 2 + (k % 2) * (n // 2)) % n


async def open_page(page, path, meshes):
    """Load an instrumented page, wait until it is ready, then stop its animation loop."""
    await page.goto(Path(path).as_uri() + "?pause=1&hud=0&scale=0.5" + ("&meshes=1" if meshes else ""))
    await page.wait_for_function("window.__hog && (__hog.error || __hog.ready)", timeout=300000)
    error = await page.evaluate("__hog.error || null")
    if error:
        raise RuntimeError(f"{Path(path).name}: {error}")
    await page.evaluate("__probe.stop()")


async def block(page, settings, frames, attempts=3):
    """Render `frames` at `settings` from a fresh history, refusing any result that is not exactly that work. A block
    whose GPU still had work after it reported completion (another process holding the GPU, or a fence that lied) is
    run again; a probe that never finishes its own work fails every attempt."""
    await page.bring_to_front()
    await page.evaluate("s => __probe.settings(s)", settings)
    for attempt in range(attempts):
        result = await page.evaluate("([f, n]) => __probe.block(f, n)", [frames, IN_FLIGHT])
        shot = result["snapshot"]
        if (result["rendered"] != len(frames) or result["loopFrames"] or shot["visible"] != "visible"
                or any(shot[k] != v for k, v in settings.items())):
            raise RuntimeError(f"probe integrity: asked for {len(frames)} frames at {settings}, got {result}")
        if result["drainMs"] <= max(DRAIN_MS, 0.5 * statistics.median(result["batches"])):
            return {**result, "retries": attempt}
    raise RuntimeError(f"GPU work outlasted the block's reported completion {attempts} times: {result}")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_times():
    """{pid: (parent pid, create time, lower-case image name, CPU seconds)} of every process, from one Windows
    NtQuerySystemInformation(SystemProcessInformation) call: one instant, and milliseconds where opening every process
    takes seconds."""
    ntdll, size = ctypes.WinDLL("ntdll"), ctypes.c_ulong(1 << 20)
    while True:
        buffer = ctypes.create_string_buffer(size.value)
        status = ntdll.NtQuerySystemInformation(5, buffer, size, ctypes.byref(size)) & 0xFFFFFFFF
        if status != 0xC0000004:                     # STATUS_INFO_LENGTH_MISMATCH: grow and ask again
            break
        size.value += 1 << 16
    if status:
        raise OSError(f"NtQuerySystemInformation failed: {status:#x}")
    out, offset, raw = {}, 0, buffer.raw
    while True:                                      # SYSTEM_PROCESS_INFORMATION entries, 64-bit layout
        step, = struct.unpack_from("<I", raw, offset)
        created, user, kernel = struct.unpack_from("<qqq", raw, offset + 32)
        length, = struct.unpack_from("<H", raw, offset + 56)
        name_at, = struct.unpack_from("<Q", raw, offset + 64)
        pid, parent = struct.unpack_from("<QQ", raw, offset + 80)
        name = ctypes.wstring_at(name_at, length // 2) if name_at else ""
        out[pid] = (parent, created, name.lower(), (user + kernel) / 1e7)
        if not step:
            return out
        offset += step


def cpu_seconds():
    """(now, the machine's busy CPU seconds, the CPU seconds of this process and every process it started, and {pid:
    CPU seconds} of the INDUCED processes outside them). The machine's interrupt and DPC time is left out: the probe's
    own GPU work causes much of it."""
    t = psutil.cpu_times()
    busy = t.user + t.system - getattr(t, "interrupt", 0.0) - getattr(t, "dpc", 0.0)
    procs, children = process_times(), {}
    for pid, (parent, created, _, _) in procs.items():
        children.setdefault(parent, []).append(pid)
    tree, todo = set(), [os.getpid()]
    while todo:                                      # a child begins after its parent, which a reused pid does not
        pid = todo.pop()
        tree.add(pid)
        todo += [c for c in children.get(pid, ()) if c not in tree and c != pid and procs[c][1] >= procs[pid][1]]
    mine = sum(procs[pid][3] for pid in tree if pid in procs)
    induced = {pid: cpu for pid, (_, _, name, cpu) in procs.items() if name in INDUCED and pid not in tree}
    return time.perf_counter(), busy, mine, induced


def loads(before, after):
    """(other, induced): the cores that the INDUCED processes present in both cpu_seconds() samples kept busy on average
    between them, and that every other process outside the probe's did."""
    (t0, busy0, mine0, listed0), (t1, busy1, mine1, listed1) = before, after
    seconds = max(t1 - t0, 1e-9)
    induced = sum(max(0.0, listed1[pid] - listed0[pid]) for pid in listed0.keys() & listed1.keys())
    return max(0.0, (busy1 - busy0) - (mine1 - mine0) - induced) / seconds, induced / seconds


def over(load, caps):
    return load[0] > caps[0] or load[1] > caps[1]


class Loaded(RuntimeError):
    """Other processes kept the machine too busy for the probe to start."""


def launch_options(gpu, profile):
    """launch_persistent_context's options: every launch reuses `profile`, writes no caches and makes no background
    traffic, leaving the security processes little of the probe's to scan or inspect."""
    return {"user_data_dir": profile, "channel": "msedge", "headless": True, "chromium_sandbox": True,
            "args": GPU_ARGS[gpu] + QUIET_ARGS, "viewport": VIEWPORT, "device_scale_factor": 1, "offline": True}


async def starting_load(caps=(MAX_OTHER_CORES, MAX_INDUCED_CORES), attempts=START_ATTEMPTS, seconds=REFERENCE_S,
                        sample=cpu_seconds):
    """Other and induced processes' busy cores before the browsers start, each measurement over `seconds`, until both
    fall within their caps or `attempts` measurements have not; the last measurement, and whether it is within."""
    for attempt in range(attempts):
        before = sample()
        await asyncio.sleep(seconds)
        load = loads(before, sample())
        if not over(load, caps):
            return load, True
        print(f"other processes keep {load[0]:.2f} cores busy, induced {load[1]:.2f}, over the caps of {caps[0]:.2f} "
              f"and {caps[1]:.2f}: waiting ({attempt + 1}/{attempts})", flush=True)
    return load, False


async def measured(window, session, measure, caps=(MAX_OTHER_CORES, MAX_INDUCED_CORES), attempts=LOAD_ATTEMPTS,
                   sample=cpu_seconds):
    """The blocks of `measure()` for one window in one session, measured again while other processes kept more than
    `caps[0]` cores busy or the induced ones more than `caps[1]`; and a load record for each attempt. The last attempt's
    blocks are kept even when contended."""
    records = []
    for attempt in range(attempts):
        before = sample()
        blocks = await measure()
        load = loads(before, sample())
        busy = over(load, caps)
        records.append({"window": window, "session": session, "attempt": attempt, "otherCores": load[0],
                        "inducedCores": load[1], "retaken": busy and attempt + 1 < attempts})
        if not busy:
            return blocks, records
    records[-1]["contended"] = True
    return blocks, records


def row(result, **fields):
    """A timed block's record: `fields` and its frame time, the mean of its scored batches, with what else it timed."""
    if result["snapshot"]["output"] != [VIEWPORT["width"], VIEWPORT["height"]]:
        raise RuntimeError(f"output {result['snapshot']['output']} is not {VIEWPORT}")
    scored = result["batches"][PREROLL // IN_FLIGHT:]
    return {**fields, "frameMs": sum(scored) / COUNT, "firstBatchMs": result["batches"][0],
            "drainMs": result["drainMs"], "retries": result["retries"], "batches": result["batches"],
            "snapshot": result["snapshot"]}


def print_window(label, rows, records):
    print(f"{label}:", "  ".join(f"{b['config']} {b['frameMs']:.1f}" for b in rows),
          f"ms; other processes {records[-1]['otherCores']:.2f} cores, induced {records[-1]['inducedCores']:.2f}",
          f"(attempt {len(records)})" if len(records) > 1 else "",
          "CONTENDED" if records[-1].get("contended") else "", flush=True)


async def open_role(context, gpu, path, role, errors):
    """A new page in `context` with `role`'s page open and checked to run on `gpu`, and its load record."""
    page = await context.new_page()
    page.on("pageerror", lambda error: errors.append(f"{role}: {error}"))
    started = time.perf_counter()
    await open_page(page, path, meshes=role == "candidate")
    adapter = await page.evaluate("document.getElementById('gpu-info').textContent")
    if gpu not in adapter.lower():
        raise RuntimeError(f"{role} rendered on {adapter}, not the {gpu} GPU")
    return page, {"page": role, "seconds": time.perf_counter() - started, "adapter": adapter,
                  "compileMs": await page.evaluate("__hog.compileMs")}


async def run_pages(playwright, gpu, paths, windows, rounds, sessions, caps, profile, errors):
    """The two-page layout: each browser session opens both pages in its load order, and each of its rounds runs a row
    of the Williams square over the configurations, the two pages' blocks interleaved."""
    blocks, other, loads_ = [], [], []
    names, orders = list(CONFIGS), balanced_orders(len(CONFIGS))
    for session in range(sessions):
        o = session % len(LOAD_ORDERS)
        context = await playwright.chromium.launch_persistent_context(**launch_options(gpu, profile))
        pages = {}
        for role in LOAD_ORDERS[o]:
            pages[role], load = await open_role(context, gpu, paths[role], role, errors)
            loads_.append({"session": session, **load})
        for window in windows:
            frames = schedule(WINDOWS[window], window in PANNED)

            async def measure(window=window, frames=frames, session=session, o=o, pages=pages):
                rows = []
                for k, r in enumerate(range(session, rounds, sessions)):
                    for position, c in enumerate(orders[session_order(session, k, len(orders))]):
                        role, meshes, dbg = CONFIGS[names[c]]
                        result = await block(pages[role], {**SETTINGS, "dbg": dbg, "meshes": meshes}, frames)
                        rows.append(row(result, window=window, round=r, session=session, order=o, kind="pair",
                                        position=position, config=names[c]))
                return rows
            rows, records = await measured(window, session, measure, caps)
            blocks += rows
            other += records
            print_window(f"session {session} {window}", rows, records)
        await context.close()
    return blocks, other, loads_


def fresh_plan(sessions, aa_sessions):
    """The fresh layout's sessions in order, as (kind, load order): pair sessions alternating the load orders, and the
    A/A sessions, ("aa", None), spread evenly among them, one after every sessions // aa_sessions pair sessions."""
    if not 1 <= aa_sessions <= sessions:
        raise ValueError(f"{aa_sessions} A/A sessions among {sessions}: a fresh run needs 1 to {sessions}")
    plan, every = [], sessions // aa_sessions
    for i in range(sessions):
        plan.append(("pair", i % len(LOAD_ORDERS)))
        if (i + 1) % every == 0 and sum(kind == "aa" for kind, _ in plan) < aa_sessions:
            plan.append(("aa", None))
    return plan


async def run_fresh(windows, rounds, sessions, aa_sessions, visit, caps=(MAX_OTHER_CORES, MAX_INDUCED_CORES),
                    sample=cpu_seconds, split=False):
    """The fresh layout: one browser at a time, a fresh one for every visit to a page. Each round of a pair session
    visits the two pages in the session's load order, and each round of an A/A session the candidate twice. A visit
    times every window after a warm-up block, which the page's first blocks need: the baseline's configuration, or the
    candidate's in the order the round's row of the Williams square gives them; an A/A visit, the integrated scene.
    `visit(role)` is an async context manager, a fresh browser with that page open, giving (time, load): `await
    time(settings, frames)` is a block's result (block()), and `load` the page's load record. With `split` each visit
    times a single window, its round's two visits to each window following each other: a window's blocks for the two
    pages then run seconds apart, not the minute a visit to every window takes, over which AMD's level drifts."""
    blocks, other, loads_ = [], [], []
    names, orders, per = list(CONFIGS), balanced_orders(len(CONFIGS)), rounds // sessions
    for session, (kind, o) in enumerate(fresh_plan(sessions, aa_sessions)):
        roles = ("candidate", "candidate") if kind == "aa" else LOAD_ORDERS[o]
        for k in range(per):
            r, williams = session * per + k, [names[c] for c in orders[session_order(session, k, len(orders))]]
            for group, (v, role) in ((g, vr) for g in ([[w] for w in windows] if split else [windows])
                                     for vr in enumerate(roles)):
                mine = ["integrated"] if kind == "aa" else [c for c in williams if CONFIGS[c][0] == role]
                async with visit(role) as (time_block, load):
                    loads_.append({"session": session, "round": r, "visit": v, "windows": group, **load})
                    for window in group:
                        frames = schedule(WINDOWS[window], window in PANNED)

                        async def measure(window=window, frames=frames, r=r, session=session, o=o, kind=kind, v=v,
                                          mine=mine, time_block=time_block):
                            settings = [{**SETTINGS, "dbg": CONFIGS[c][2], "meshes": CONFIGS[c][1]} for c in mine]
                            await time_block(settings[0], frames)
                            rows = []
                            for position, (c, s) in enumerate(zip(mine, settings)):
                                rows.append(row(await time_block(s, frames), window=window, round=r, session=session,
                                                order=o, kind=kind, visit=v, position=position, config=c))
                            return rows
                        rows, records = await measured(window, session, measure, caps, sample=sample)
                        blocks += rows
                        other += records
                        print_window(f"session {session} round {r} {role} {window}", rows, records)
    return blocks, other, loads_


def fresh_visits(playwright, gpu, paths, profile, errors):
    """run_fresh's visits: each launches a browser on `gpu` with `profile`, opens the role's page, and closes the
    browser after."""
    @contextlib.asynccontextmanager
    async def visit(role):
        context = await playwright.chromium.launch_persistent_context(**launch_options(gpu, profile))
        try:
            page, load = await open_role(context, gpu, paths[role], role, errors)
            yield (lambda settings, frames: block(page, settings, frames)), load
        finally:
            await context.close()
    return visit


async def run(gpu, baseline, candidate, windows, rounds, caps=(MAX_OTHER_CORES, MAX_INDUCED_CORES), profile=None,
              sessions=SESSIONS, layout="pages", aa_sessions=AA_SESSIONS):
    sources, errors = {"baseline": baseline, "candidate": candidate}, []
    start, settled = await starting_load(caps)
    if not settled:
        raise Loaded(f"other processes kept {start[0]:.2f} cores busy and the induced ones {start[1]:.2f}, over the "
                     f"caps of {caps[0]:.2f} and {caps[1]:.2f}, for {START_ATTEMPTS * REFERENCE_S:.0f} s before the "
                     "browsers were to start")
    print(f"before the browsers start: other processes {start[0]:.2f} cores, induced {start[1]:.2f}; a window over "
          f"{caps[0]:.2f} or {caps[1]:.2f} is measured again", flush=True)
    with tempfile.TemporaryDirectory() as folder:
        paths = {role: Path(folder) / f"{role}.html" for role in sources}
        for role, source in sources.items():
            instrument(source, paths[role])
        profile = profile or Path(folder) / "profile"
        async with async_playwright() as playwright:
            if layout in ("fresh", "fresh-window"):
                blocks, other, loads_ = await run_fresh(windows, rounds, sessions, aa_sessions,
                                                        fresh_visits(playwright, gpu, paths, profile, errors), caps,
                                                        split=layout == "fresh-window")
            else:
                blocks, other, loads_ = await run_pages(playwright, gpu, paths, windows, rounds, sessions, caps,
                                                        profile, errors)
    if errors:
        raise RuntimeError(f"page errors: {errors}")
    return {"gpu": gpu, "layout": layout, "viewport": VIEWPORT, "dpr": 1, "settings": SETTINGS, "inFlight": IN_FLIGHT,
            "preroll": PREROLL, "count": COUNT, "fps": FPS, "pan": PAN, "windows": {w: WINDOWS[w] for w in windows},
            "configs": CONFIGS, "rounds": rounds, "sessions": sessions,
            "aaSessions": aa_sessions if layout != "pages" else 0, "orders": balanced_orders(len(CONFIGS)),
            "loadOrders": LOAD_ORDERS, "loads": loads_,
            "pages": {role: {"path": str(source), "sha256": digest(source)} for role, source in sources.items()},
            "clock": "performance.now from submission to an observed fence, per batch of two frames",
            "otherLoad": {"startCores": start[0], "startInducedCores": start[1], "capCores": caps[0],
                          "inducedCapCores": caps[1], "attempts": LOAD_ATTEMPTS, "windows": other},
            "blocks": blocks, "summary": summarize(blocks, other, fresh=layout != "pages")}


def geomean(values):
    return math.exp(statistics.fmean(map(math.log, values)))


def ratio_of(num, den, order=None):
    """A ratio's estimate from `groups`, one per load order, of sessions (lists of rounds, {configuration: frame ms}):
    the geometric mean of the sessions' medians of the paired ratio, over every load order, or over load order `order`
    alone."""
    return lambda groups: geomean([statistics.median(r[num] / r[den] for r in s)
                                   for o, g in enumerate(groups) if order in (None, o) for s in g])


def opened_second(page):
    """The load order whose sessions open `page` second."""
    return next(o for o, order in enumerate(LOAD_ORDERS) if order[1] == page)


def resampled(items, rng):
    return [rng.choice(items) for _ in items]


def interval(groups, statistic, rng):
    """`statistic`'s BAND interval by bootstrap over `groups`, as ratio_of takes them. Each draw takes n - 1 of a load
    order's n sessions again, which gives the variation between sessions, between browser launches, its unbiased share
    (Rao and Wu's rescaled bootstrap), and then each drawn session's rounds."""
    draws = sorted(statistic([[resampled(rng.choice(g), rng) for _ in g[1:]] for g in groups]) for _ in range(DRAWS))
    return draws[int((1 - BAND) / 2 * DRAWS)], draws[int((1 + BAND) / 2 * DRAWS) - 1]


def beyond(band, tolerance=TOLERANCE):
    """Whether an interval lies wholly further from 1 than `tolerance`, either way."""
    return band[0] > tolerance or band[1] < 1 / tolerance


def absolute_check(blocks):
    """The fresh layout's A/A, pooled over the windows: each A/A round's geometric mean over the windows of its second
    visit's frame time over its first's, each session's median of those, and their geometric mean with its BAND
    interval, drawing n - 1 of the sessions and then their rounds (no interval with fewer than MIN_SESSIONS). None for a
    run without A/A rounds."""
    rounds = {}
    for b in blocks:
        rounds.setdefault((b["session"], b["round"]), {}).setdefault(b["window"], {})[b["visit"]] = b["frameMs"]
    if not rounds:
        return None
    sessions = {}
    for (session, _), by_window in sorted(rounds.items()):
        sessions.setdefault(session, []).append(geomean([ms[1] / ms[0] for ms in by_window.values()]))
    groups = [list(sessions.values())]

    def statistic(g):
        return geomean([statistics.median(s) for s in g[0]])
    return {"estimate": statistic(groups),
            "band": interval(groups, statistic, random.Random("absolute")) if len(groups[0]) >= MIN_SESSIONS else None}


def summarize(blocks, other=(), fresh=False):
    """Per window: each configuration's median frame time; each paired ratio's estimate, BAND interval, sessions'
    medians and pairs by round, and for a ratio between the two pages its bounds; the order effect; with A/A rounds
    (the fresh layout's, kind "aa", kept out of the ratios), their check; and a verdict on integrated/baseline against
    BUDGET. A block's `order` names its session's load order (the session's parity when it has none). A `fresh` run
    needs its A/A.

    Opening a page second (in the fresh layout, visiting it second in a round) changes its frame time by a factor of its
    own: on the AMD iGPU it slowed pages by 7-34%, and
    on NVIDIA the later copy ran up to 1.4% faster, in the A/A tests. So long as both pages move the same way, a ratio
    between the two pages lies between its values in the sessions that open its numerator's page second (`upper`) and
    in those that open its denominator's page second (`lower`), however unequally the two pages are moved. A window
    passes when both bounds' intervals lie within the budget and fails when both lie above it, unless a check flags it;
    otherwise it is inconclusive.
    The order effect, the square root of the upper bound over the lower, is the geometric mean of the two pages'
    factors as the page opened second: the further it is from 1, the further apart the bounds. The checks (`flags`):
    - design: a load order has fewer than MIN_SESSIONS sessions, or a session ran fewer rounds than there are
      configurations, so the intervals cannot hold the variation between browser launches;
    - control: the control over the integrated scene, one configuration timed against itself, lies beyond TOLERANCE
      throughout its interval, so the measurement has an error of its own;
    - absolute: the A/A's interval, the candidate's second fresh visit over its first, lies beyond TOLERANCE throughout,
      a gross error in fresh visits themselves, as the control's check is one in the measurement (with fewer than
      MIN_SESSIONS A/A sessions, design). The bounds already hold any effect of visiting a page second that moves both
      pages the same way, so the verdict does not need the A/A to show the visits alike.
    Also the other and induced processes' busy cores over each session's kept attempt."""
    out, timed = {}, [b for b in blocks if b.get("kind") != "aa"]
    absolute = absolute_check([b for b in blocks if b.get("kind") == "aa"])
    if fresh and absolute is None:
        absolute = {"estimate": None, "band": None}
    for window in dict.fromkeys(b["window"] for b in timed):
        rounds, order_of = {}, {}
        for b in timed:
            if b["window"] == window:
                rounds.setdefault((b["session"], b["round"]), {})[b["config"]] = b["frameMs"]
                order_of[b["session"]] = b.get("order", b["session"] % len(LOAD_ORDERS))
        sessions = {}
        for (session, _), ms in sorted(rounds.items()):
            sessions.setdefault(session, []).append(ms)
        groups = [[s for k, s in sorted(sessions.items()) if order_of[k] == o] for o in range(len(LOAD_ORDERS))]
        designed = (all(len(g) >= MIN_SESSIONS and all(len(s) >= len(CONFIGS) for s in g) for g in groups)
                    and (absolute is None or absolute["band"] is not None))

        def estimated(statistic, name, window=window, groups=groups, designed=designed):
            return {"estimate": statistic(groups) if all(groups) else None,
                    "band": interval(groups, statistic, random.Random(f"{window}|{name}")) if designed else None}
        entry = {"frameMs": {c: statistics.median(ms[c] for ms in rounds.values()) for c in CONFIGS}}
        for num, den in RATIOS:
            pairs = [rounds[k][num] / rounds[k][den] for k in sorted(rounds, key=lambda k: k[1])]
            entry[f"{num}/{den}"] = {
                **estimated(ratio_of(num, den), f"{num}/{den}"),
                "sessions": [statistics.median(r[num] / r[den] for r in s) for _, s in sorted(sessions.items())],
                "min": min(pairs), "max": max(pairs), "pairs": pairs}
            if CONFIGS[num][0] != CONFIGS[den][0]:
                for bound, page in (("upper", CONFIGS[num][0]), ("lower", CONFIGS[den][0])):
                    entry[f"{num}/{den}"][bound] = estimated(ratio_of(num, den, opened_second(page)),
                                                             f"{num}/{den} {bound}")
        upper, lower = (ratio_of("integrated", "baseline", opened_second(page)) for page in ("candidate", "baseline"))
        entry["orderEffect"] = estimated(lambda g: math.sqrt(upper(g) / lower(g)), "order")
        flags = [] if designed else ["design"]
        if designed and beyond(entry["control/integrated"]["band"]):
            flags.append("control")
        if absolute is not None:
            entry["absolute"] = absolute
            if designed and beyond(absolute["band"]):
                flags.append("absolute")
        judged = entry["integrated/baseline"]
        bands = (judged["upper"]["band"], judged["lower"]["band"])
        entry["flags"] = flags
        entry["verdict"] = ("inconclusive" if flags else "pass" if max(b[1] for b in bands) <= BUDGET else
                            "fail" if min(b[0] for b in bands) > BUDGET else "inconclusive")
        entry["otherCores"] = [x["otherCores"] for x in other if x["window"] == window and not x["retaken"]]
        entry["inducedCores"] = [x["inducedCores"] for x in other if x["window"] == window and not x["retaken"]]
        out[window] = entry
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gpu", choices=sorted(GPU_ARGS), required=True)
    parser.add_argument("--baseline", type=Path, required=True, help="the frozen page the candidate is compared with")
    parser.add_argument("--candidate", type=Path, default=ROOT / "hogwarts-flyover.html")
    parser.add_argument("--windows", default=",".join(WINDOWS))
    parser.add_argument("--layout", choices=("pages", "fresh", "fresh-window"),
                        help="both pages open in each browser session (pages), or one fresh browser for every visit to "
                             "a page, which times every window (fresh) or one (fresh-window); default LAYOUTS")
    parser.add_argument("--rounds", type=int,
                        help="a multiple of --sessions times the configurations, so each session runs every order "
                             f"(default {ROUNDS}, fresh {FRESH_ROUNDS})")
    parser.add_argument("--sessions", type=int,
                        help=f"sessions, alternating the load orders: at least {MIN_SESSIONS} per order "
                             f"(default {SESSIONS}, fresh {FRESH_SESSIONS})")
    parser.add_argument("--aa-sessions", type=int, default=AA_SESSIONS,
                        help=f"the fresh layout's A/A sessions, which visit the candidate twice: at least "
                             f"{MIN_SESSIONS}, and no more than --sessions")
    parser.add_argument("--max-other-cores", type=float, default=MAX_OTHER_CORES,
                        help="other processes' busy cores beyond which a measurement is not kept")
    parser.add_argument("--max-induced-cores", type=float, default=MAX_INDUCED_CORES,
                        help="security and OS processes' busy cores beyond which a measurement is not kept")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(1, str(ROOT))
    from tools import locks                      # the quiet lock lives in the state folder every worktree shares
    from tools.paths import cache_dir
    locks.require_quiet("mesh_probe.py")
    windows = args.windows.split(",")
    if unknown := set(windows) - set(WINDOWS):
        parser.error(f"unknown windows {sorted(unknown)}")
    layout = args.layout or LAYOUTS[args.gpu]
    rounds = args.rounds or (ROUNDS if layout == "pages" else FRESH_ROUNDS)
    sessions = args.sessions or (SESSIONS if layout == "pages" else FRESH_SESSIONS)
    if sessions < MIN_SESSIONS * len(LOAD_ORDERS) or sessions % len(LOAD_ORDERS):
        parser.error(f"--sessions must be a multiple of {len(LOAD_ORDERS)}, at least {MIN_SESSIONS} per load order")
    if rounds < sessions * len(CONFIGS) or rounds % (sessions * len(CONFIGS)):
        parser.error(f"--rounds must be a multiple of --sessions times {len(CONFIGS)}, so each session runs every order")
    if layout != "pages" and not MIN_SESSIONS <= args.aa_sessions <= sessions:
        parser.error(f"--aa-sessions must be at least {MIN_SESSIONS} and no more than --sessions")
    caps = (args.max_other_cores, args.max_induced_cores)
    try:
        result = asyncio.run(run(args.gpu, args.baseline, args.candidate, windows, rounds, caps,
                                 cache_dir("probe-profile"), sessions, layout=layout, aa_sessions=args.aa_sessions))
    except Loaded as e:
        print(f"blocked: {e}; run it again once the machine is quiet", file=sys.stderr)
        return 3
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    def shown(v):
        if v["estimate"] is None:
            return "n/a"
        return f"{v['estimate']:.3f} " + (f"[{v['band'][0]:.3f}-{v['band'][1]:.3f}]" if v["band"] else "[no interval]")
    if absolute := next((s["absolute"] for s in result["summary"].values() if "absolute" in s), None):
        print(f"fresh visits' A/A, the candidate's second visit over its first: {shown(absolute)}; every window is "
              f"inconclusive if it lies beyond {TOLERANCE:.2f} either way throughout")
    for window, s in result["summary"].items():
        judged = s["integrated/baseline"]
        print(f"{window:14}", "  ".join(f"{n}/{d} {shown(s[f'{n}/{d}'])}" for n, d in RATIOS),
              f"  integrated/baseline from {shown(judged['lower'])} to {shown(judged['upper'])}",
              f"  order effect {shown(s['orderEffect'])}",
              f"  {s['verdict']} against {BUDGET:.2f}", f"({', '.join(s['flags'])})" if s["flags"] else "",
              "  other processes", "/".join(f"{c:.2f}" for c in s["otherCores"]),
              "cores, induced", "/".join(f"{c:.2f}" for c in s["inducedCores"]))
    if contended := sorted({(x["window"], x["session"]) for x in result["otherLoad"]["windows"] if x.get("contended")}):
        print(f"blocked: other processes kept more than {caps[0]:.2f} cores busy, or the induced ones more than "
              f"{caps[1]:.2f}, through every attempt at {contended} (window, session), so these ratios are not "
              "results; run it again once the machine is quiet", file=sys.stderr)
        return 3
    if short := {w: s["verdict"] for w, s in result["summary"].items() if s["verdict"] != "pass"}:
        print(f"not every window passes the budget of {BUDGET:.2f}: {short}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
