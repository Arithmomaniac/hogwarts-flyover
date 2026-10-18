"""Headless check: compile status, console errors, timed screenshots of the flyover."""
import asyncio, json, sys, time, os
from pathlib import Path
from playwright.async_api import async_playwright

URL = (Path(__file__).resolve().parents[1] / "hogwarts-flyover.html").as_uri()
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "view", "shots")


async def main():
    times = [float(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["0"])]
    extra = sys.argv[2] if len(sys.argv) > 2 else ""
    tag = sys.argv[3] if len(sys.argv) > 3 else "s"
    os.makedirs(OUTDIR, exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="msedge", headless=True,
                                    args=["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"])
        pg = await b.new_page(viewport={"width": 1280, "height": 720})
        logs = []
        pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
        pg.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
        t0 = time.time()
        await pg.goto(URL + f"?pause=1&t={times[0]}&scale=0.5&q=1{extra}")
        st = None
        for _ in range(600):
            st = await pg.evaluate("window.__hog ? JSON.stringify({r: __hog.ready, e: __hog.error || null, f: __hog.frames, T: __hog.total, c: __hog.compileMs, pc: __hog.pcomp}) : null")
            if st:
                d = json.loads(st)
                if d["e"] or d["r"]:
                    break
            await asyncio.sleep(0.5)
        print("status", st, f"after {time.time() - t0:.1f}s")
        msg = await pg.evaluate("document.getElementById('loadMsg').textContent")
        print("loadMsg:", msg[:3000])
        for l in logs[:40]:
            print(l[:2000])
        d = json.loads(st) if st else {}
        if d.get("r"):
            for t in times:
                await pg.evaluate(f"__hog.set({t})")
                f0 = await pg.evaluate("__hog.frames")
                t1 = time.time()
                while await pg.evaluate("__hog.frames") < f0 + 12:
                    await asyncio.sleep(0.05)
                dtf = (time.time() - t1) / 12
                fn = os.path.join(OUTDIR, f"{tag}_{int(t):04d}.png")
                await pg.screenshot(path=fn)
                print(f"t={t}: ~{dtf * 1000:.0f} ms/frame ->", fn)
        await b.close()


asyncio.run(main())
