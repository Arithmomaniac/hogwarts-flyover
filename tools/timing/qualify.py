"""Frame-time qualification in one quiet window, unattended: NVIDIA on two pages, a trial of AMD's fresh-window layout,
then AMD on the layout that trial supports, all in one hold of the quiet lock.

NVIDIA runs first, on the probe's default two-page layout, so its verdicts stand even if the window ends early. After a
pause to let the machine cool, the trial times TRIAL_WINDOWS on AMD with fresh-window at the qualification's smallest
design (TRIAL). From the trial's rounds the driver estimates the paired ratio's scatter between rounds and between
sessions (scatter), and projects how wide a bound's interval comes out at a design by simulating the probe's own
bootstrap (halfwidth). When the trial's design projects bounds within LIMIT and the trial's A/A raises no flag, AMD runs
fresh-window on every window at that design, the smallest the qualification takes (16 pair rounds), with the FOCUS
windows, those nearest the budget, measured again in a run of their own whose rounds join the first run's. If the two
runs would outlast --max-amd-hours at the trial's pace, the focus run is left out and the FOCUS windows are judged on
the first run alone. Otherwise, as when the trial measured nothing, or when the first run alone would outlast
--max-amd-hours, AMD runs on two pages, judged by its bounds. A step that other processes' load blocks (exit status 3)
runs again after BLOCKED_WAIT_S, up to BLOCKED_ATTEMPTS times; a step that measured never runs again. The outcome is
NVIDIA's and AMD's verdicts, never the trial's. Each step's log and probe report, and summary.json, go to a folder of
the run's own, build/logs/qualify-PAGEHASH-STARTED/, so no run overwrites another's.

    python -X utf8 -m tools.timing.qualify [--max-amd-hours 2] [--dry-run]
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

from tools import inputs, locks, stages
from tools.paths import LOGS, PAGE, ROOT, page_tag
from tools.site import baseline as frozen
import mesh_probe as mp

TRIAL_WINDOWS = ("castle", "trees-65", "greenhouses")
TRIAL = {"rounds": 16, "sessions": 4, "aa_sessions": 2}     # two sessions of four rounds per load order; two A/A
TRIAL_STEP = "trial-amd"
ROUNDS_PER_SESSION = TRIAL["rounds"] // TRIAL["sessions"]   # every row of the Williams square, as the probe requires
LIMIT = 0.05                     # half a bound's interval, in log terms: about ±5%
FOCUS = ("greenhouses",)         # the window nearest the budget, measured twice as long
PAGES = {"layout": "pages", "runs": [{"name": "pages", "args": {}}]}
MAX_AMD_HOURS = 2.0
MEDIAN_VARIANCE = {4: 1.19, 8: 1.35}     # the variance of the median of n normal draws over that of their mean
REPLICATES = 24                  # simulated runs per projected design
COOL_S = 120                     # between NVIDIA's run and AMD's, which share the laptop's cooling
BLOCKED, BLOCKED_WAIT_S, BLOCKED_ATTEMPTS = 3, 300, 3
SESSION_STRIDE = 1000            # numbers one run's sessions and rounds apart from another's when they are merged


def probe_argv(gpu, out, baseline, layout=None, windows=None, rounds=None, sessions=None, aa_sessions=None):
    argv = ["src/mesh_probe.py", "--gpu", gpu, "--baseline", baseline, "--out", out]
    if layout:
        argv += ["--layout", layout]
    if windows:
        argv += ["--windows", ",".join(windows)]
    for flag, value in (("--rounds", rounds), ("--sessions", sessions), ("--aa-sessions", aa_sessions)):
        if value is not None:
            argv += [flag, str(value)]
    return argv


def step(name, argv, token, folder, runner=stages.python, sleep=time.sleep):
    """mesh_probe with `argv` under the quiet lock's `token`, run again after BLOCKED_WAIT_S while other processes' load
    blocks it: its name, exit status, attempts, the last attempt's seconds, log and report (None unless it measured)."""
    out, log = Path(argv[argv.index("--out") + 1]), folder / f"{name}.log"
    for attempt in range(1, BLOCKED_ATTEMPTS + 1):
        out.unlink(missing_ok=True)
        started = time.monotonic()
        status = runner(*argv, log=log, env={locks.TOKEN: token})
        seconds = time.monotonic() - started
        if status != BLOCKED or attempt == BLOCKED_ATTEMPTS:
            break
        sleep(BLOCKED_WAIT_S)
    result = json.loads(out.read_text(encoding="utf-8")) if status in (0, 1) and out.is_file() else None
    return {"name": name, "status": status, "attempts": attempt, "seconds": seconds, "log": str(log), "result": result}


def scatter(result):
    """integrated/baseline's scatter in a fresh run's report, in log terms: `round`, between the rounds of a session,
    from the median absolute difference of every two of a session's rounds in a window (robust to the odd slow visit);
    and `session`, between the sessions of a load order, beyond what `round` explains."""
    rounds, sessions = {}, {}
    for b in result["blocks"]:
        if b.get("kind") == "pair" and b["config"] in ("integrated", "baseline"):
            rounds.setdefault((b["window"], b["order"], b["session"], b["round"]), {})[b["config"]] = b["frameMs"]
    for (window, order, session, _), ms in sorted(rounds.items()):
        sessions.setdefault((window, order, session), []).append(math.log(ms["integrated"] / ms["baseline"]))
    differences = [abs(a - b) for xs in sessions.values() for i, a in enumerate(xs) for b in xs[i + 1:]]
    sigma_round = statistics.median(differences) / (math.sqrt(2) * statistics.NormalDist().inv_cdf(0.75))
    per = statistics.median(len(xs) for xs in sessions.values())
    groups = {}
    for (window, order, _), xs in sessions.items():
        groups.setdefault((window, order), []).append(statistics.median(xs))
    within = [(len(ms) - 1, statistics.variance(ms)) for ms in groups.values() if len(ms) > 1]
    pooled = sum(df * v for df, v in within) / sum(df for df, _ in within)
    explained = MEDIAN_VARIANCE.get(per, math.pi / 2) * sigma_round ** 2 / per
    return {"round": sigma_round, "session": math.sqrt(max(0.0, pooled - explained)), "roundsPerSession": per}


def halfwidth(sigma_round, sigma_session, sessions, rounds=ROUNDS_PER_SESSION, replicates=REPLICATES, seed=0):
    """Half the width, in log terms, of the interval mesh_probe.interval gives a bound measured over `sessions` sessions
    of `rounds` rounds, with normal scatter `sigma_round` between rounds and `sigma_session` between sessions: the
    median over `replicates` simulated runs."""
    rng, widths, statistic = random.Random(seed), [], mp.ratio_of("integrated", "baseline")
    for _ in range(replicates):
        group = []
        for _ in range(sessions):
            offset = rng.gauss(0.0, sigma_session)
            group.append([{"integrated": math.exp(offset + rng.gauss(0.0, sigma_round)), "baseline": 1.0}
                          for _ in range(rounds)])
        low, high = mp.interval([group], statistic, random.Random(rng.random()))
        widths.append(math.log(high / low) / 2)
    return statistics.median(widths)


def visits(run):
    """How many fresh browsers a fresh-window run opens: two per round, window and A/A or pair round."""
    return (run["rounds"] + ROUNDS_PER_SESSION * run["aa_sessions"]) * len(run["windows"]) * 2


def amd_design(stats, aa_flagged, visit_s, max_hours, projection=halfwidth):
    """AMD's runs, from the trial's scatter (`stats`), whether its A/A was flagged and its seconds per visit:
    fresh-window at the trial's design, with the FOCUS windows run again, when the trial supports it and both runs fit
    in `max_hours`; without the FOCUS run when only the first fits; else two pages. Each run is a name and the probe's
    arguments; with the projected seconds and the reasons for the layout."""
    trial = projection(stats["round"], stats["session"], TRIAL["sessions"] // 2)
    design = {"trialHalfwidth": trial, "reasons": []}
    if trial > LIMIT:
        design["reasons"].append(f"the trial's design projects bounds of ±{trial:.1%}, over ±{LIMIT:.0%}")
    if aa_flagged:
        design["reasons"].append("the trial's A/A shows a difference beyond the probe's tolerance")
    if not design["reasons"]:
        runs = [{"name": "fresh", "args": {"layout": "fresh-window", "windows": tuple(mp.WINDOWS), **TRIAL}},
                {"name": "focus", "args": {"layout": "fresh-window", "windows": FOCUS, **TRIAL}}]
        seconds = [visits(run["args"]) * visit_s for run in runs]
        if sum(seconds) <= max_hours * 3600:
            return {**design, "layout": "fresh-window", "runs": runs, "projectedS": sum(seconds)}
        if seconds[0] <= max_hours * 3600:
            design["reasons"].append(f"the focus run would take the runs to {sum(seconds) / 3600:.1f} h")
            return {**design, "layout": "fresh-window", "runs": runs[:1], "projectedS": seconds[0]}
        design["reasons"].append(f"the fresh-window run projects {seconds[0] / 3600:.1f} h, over {max_hours:g} h")
    return {**design, **PAGES}


def merged(results, window):
    """`window`'s summary from fresh runs' reports taken together: their pair blocks in the window and every A/A block,
    each run's sessions and rounds numbered apart."""
    blocks, other = [], []
    for k, result in enumerate(results):
        shift = k * SESSION_STRIDE
        blocks += [{**b, "session": b["session"] + shift, "round": b["round"] + shift} for b in result["blocks"]
                   if b["window"] == window or b.get("kind") == "aa"]
        other += [{**x, "session": x["session"] + shift} for x in result["otherLoad"]["windows"]
                  if x["window"] == window]
    return mp.summarize(blocks, other, fresh=True)[window]


def verdicts(steps, focus=FOCUS):
    """{window: summary} from a GPU's measured steps: one run's own summaries, or, with a focus run, the FOCUS windows'
    from both runs' rounds together."""
    results = [s["result"] for s in steps]
    out = dict(results[0]["summary"])
    if len(results) > 1:
        out.update({window: merged(results, window) for window in focus})
    return out


def outcome(steps, judged):
    """The qualification's outcome from its judged steps (NVIDIA's and AMD's, not the trial's) and their verdicts."""
    if any(s["status"] == BLOCKED for s in steps):
        return "blocked"
    if any(s["result"] is None for s in steps):
        return "error"
    return "pass" if all(v["verdict"] == "pass" for gpu in judged.values() for v in gpu.values()) else "fail"


def qualify(baseline, folder, max_hours, quiet=locks.quiet, run=step, sleep=time.sleep, projection=halfwidth,
            say=print):
    """The whole qualification under one hold of the quiet lock: its summary. A trial that measured nothing (blocked,
    or crashed) sends AMD to two pages, as a noisy one does."""
    steps, judged, trial_stats, design = [], {}, None, None
    with quiet(f"qualification in {ROOT.name}", say) as token:
        nvidia = run("nvidia", probe_argv("nvidia", folder / "nvidia.json", baseline), token, folder)
        steps.append(nvidia)
        if nvidia["result"] is not None:
            judged["nvidia"] = verdicts([nvidia])
        sleep(COOL_S)
        trial = run(TRIAL_STEP, probe_argv("amd", folder / f"{TRIAL_STEP}.json", baseline, layout="fresh-window",
                                           windows=TRIAL_WINDOWS, **TRIAL), token, folder)
        steps.append(trial)
        if trial["result"] is not None:
            trial_stats = {**scatter(trial["result"]), "visitS": trial["seconds"] / len(trial["result"]["loads"]),
                           "absolute": next(iter(trial["result"]["summary"].values())).get("absolute")}
            aa_flagged = any("absolute" in s["flags"] for s in trial["result"]["summary"].values())
            design = amd_design(trial_stats, aa_flagged, trial_stats["visitS"], max_hours, projection)
        else:
            why = "other processes' load blocked it" if trial["status"] == BLOCKED else \
                f"it exited with status {trial['status']} and no report"
            design = {"trialHalfwidth": None, "reasons": [f"the trial measured nothing: {why}"], **PAGES}
        say(f"AMD runs {design['layout']}" + (f": {'; '.join(design['reasons'])}" if design["reasons"] else ""))
        amd = [run(f"amd-{r['name']}", probe_argv("amd", folder / f"amd-{r['name']}.json", baseline, **r["args"]),
                   token, folder) for r in design["runs"]]
        steps += amd
        if all(s["result"] is not None for s in amd):
            judged["amd"] = verdicts(amd)
    return {"page": page_tag(), "head": git_head(), "baseline": inputs.spec("baseline-x")["sha256"],
            "rules": {"trialWindows": TRIAL_WINDOWS, "trial": TRIAL, "limit": LIMIT, "focus": FOCUS,
                      "maxAmdHours": max_hours},
            "steps": [{k: v for k, v in s.items() if k != "result"} for s in steps], "trial": trial_stats,
            "design": design, "verdicts": judged,
            "outcome": outcome([s for s in steps if s["name"] != TRIAL_STEP], judged)}


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None


def spread(cores):
    """A window's busy cores over its sessions, lowest to highest."""
    return f"{min(cores):.1f}-{max(cores):.1f}" if cores else "n/a"


def report(summary):
    lines = [f"qualification of {summary['page']} at {summary['head']}: {summary['outcome'].upper()}"]
    for s in summary["steps"]:
        lines.append(f"  {s['name']:12} exit {s['status']} after {s['attempts']} attempt(s), "
                     f"{s['seconds'] / 60:.0f} min")
    if summary["trial"]:
        t = summary["trial"]
        lines.append(f"  trial: scatter between rounds {t['round']:.3f}, between sessions {t['session']:.3f} (log), "
                     f"{t['visitS']:.1f} s a visit")
    if summary["design"]:
        d = summary["design"]
        projected = f"; the trial's design projected bounds of ±{d['trialHalfwidth']:.1%}" \
            if d["trialHalfwidth"] is not None else ""
        lines.append(f"  AMD on {d['layout']}" + (f" ({'; '.join(d['reasons'])})" if d["reasons"] else "") + projected)
    for gpu, windows in summary["verdicts"].items():
        for window, v in windows.items():
            judged = v["integrated/baseline"]
            bounds = " to ".join(f"{b['estimate']:.3f} [{b['band'][0]:.3f}-{b['band'][1]:.3f}]" if b["band"] else "n/a"
                                 for b in (judged["lower"], judged["upper"]))
            lines.append(f"  {gpu:6} {window:14} {v['verdict']:12} integrated/baseline from {bounds}"
                         + (f" ({', '.join(v['flags'])})" if v["flags"] else "")
                         + f"; induced {spread(v.get('inducedCores'))} cores, other {spread(v.get('otherCores'))}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tools.timing.qualify", description=__doc__.splitlines()[0])
    parser.add_argument("--max-amd-hours", type=float, default=MAX_AMD_HOURS,
                        help="the longest AMD's fresh-window runs may project to take: beyond it the focus run is left "
                             "out, and beyond it for the first run alone AMD runs on two pages")
    parser.add_argument("--dry-run", action="store_true", help="print the steps, measure nothing")
    args = parser.parse_args(argv)
    try:
        baseline = inputs.require("baseline-x")
        frozen.check_baseline(baseline, PAGE)
    except inputs.Blocked as e:
        print(f"blocked: {e}", file=sys.stderr)
        return BLOCKED
    started = datetime.datetime.now()
    folder = LOGS / f"qualify-{page_tag()}-{started:%Y%m%d-%H%M%S}"
    if args.dry_run:
        trial = probe_argv("amd", folder / f"{TRIAL_STEP}.json", baseline, layout="fresh-window",
                           windows=TRIAL_WINDOWS, **TRIAL)
        print("1. NVIDIA, two pages:", *probe_argv("nvidia", folder / "nvidia.json", baseline))
        print(f"2. {COOL_S} s to cool, then the AMD trial, fresh-window:", *trial)
        print(f"3. AMD on fresh-window at the trial's design, every window and again {', '.join(FOCUS)}, when the "
              f"trial projects bounds within ±{LIMIT:.0%} and its A/A is not flagged, if the runs fit in "
              f"{args.max_amd_hours:g} h (without the {', '.join(FOCUS)} run if only the first fits); "
              "else on two pages")
        return 0
    folder.mkdir(parents=True)
    summary = {"started": started.isoformat(timespec="seconds"), **qualify(baseline, folder, args.max_amd_hours),
               "ended": datetime.datetime.now().isoformat(timespec="seconds")}
    (folder / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    text = report(summary)
    (folder / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text, f"reports in {folder}", sep="\n")
    return {"pass": 0, "fail": 1, "error": 1, "blocked": BLOCKED}[summary["outcome"]]


if __name__ == "__main__":
    sys.exit(main())
