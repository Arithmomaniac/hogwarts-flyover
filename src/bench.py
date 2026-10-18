"""Benchmark frame time per debug configuration (one page load, one compile)."""
import asyncio, json, sys, time, os
from pathlib import Path
from playwright.async_api import async_playwright

URL = (Path(__file__).resolve().parents[1] / "hogwarts-flyover.html").as_uri()
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "view", "shots")


async def frame_ms(pg, n=4):
    f0 = await pg.evaluate("__hog.frames")
    while await pg.evaluate("__hog.frames") < f0 + 1:
        await asyncio.sleep(0.02)
    t1 = time.time(); f1 = await pg.evaluate("__hog.frames")
    while await pg.evaluate("__hog.frames") < f1 + n:
        await asyncio.sleep(0.02)
    return (time.time() - t1) / n * 1000


async def main():
    times = [float(x) for x in sys.argv[1].split(",")]
    cfgs = [int(x) for x in sys.argv[2].split(",")]
    shots = sys.argv[3].split(",") if len(sys.argv) > 3 else []
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="msedge", headless=True,
                                    args=["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"])
        pg = await b.new_page(viewport={"width": 1280, "height": 720})
        t0 = time.time()
        await pg.goto(URL + "?pause=1&t=0&scale=0.5&q=1&hud=0")
        for _ in range(1200):
            st = await pg.evaluate("window.__hog ? JSON.stringify({r: __hog.ready, e: __hog.error || null, c: __hog.compileMs}) : null")
            if st and (json.loads(st)["r"] or json.loads(st)["e"]):
                break
            await asyncio.sleep(0.5)
        print("ready", st, f"{time.time() - t0:.0f}s")
        for t in times:
            await pg.evaluate(f"__hog.set({t})")
            row = []
            for c in cfgs:
                await pg.evaluate(f"__hog.dbg({c})")
                ms = await frame_ms(pg)
                row.append(f"{c}:{ms:.0f}")
                if f"{int(t)}/{c}" in shots:
                    await pg.screenshot(path=os.path.join(OUTDIR, f"dbg_{int(t)}_{c}.png"))
            print(f"t={t}: " + "  ".join(row))
        await b.close()


asyncio.run(main())
