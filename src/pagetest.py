"""One browser per test process, for the browser suites, and one page per PageTestCase class or per test_ui test.

Opening the page dominates a browser test: on NVIDIA, Edge takes about 90 s to compile its programs, and only a running
browser keeps them (a second context in it opens the page in about 5 s, while a fresh browser on a profile kept on disk
still took 75-83 s). So the EdgeTestCase classes of a process share one Edge (EDGE), launched by the first that needs it
and closed at exit, or by tools.testrun before it lets the GPU go. A PageTestCase class opens its own context and page
in it, in setUpClass, and closes them in tearDownClass; test_ui's tests, which change more of the page than the hooks
below restore, each open a context and page of their own, in setUp, and close them in tearDown. A fixture that fails
closes the browser too, so the next one launches a fresh browser. Before each test a PageTestCase class restores,
through the page's __hog hooks, what tests change: the viewport, the time of day, the debug bits, meshes on with the
embedded payload, and the opening view. A page that has died, or whose frame loop has stopped, is reopened. Test
methods are coroutines, run on the browser's event loop.

HOG_GPU=amd|nvidia pins Edge to that adapter, as the dual-GPU timing harness does; when the page reports another adapter
every test of the class is skipped as "blocked:", which tools.testrun reports as blocked, never as a pass.
"""
import asyncio
import atexit
import contextlib
import functools
import inspect
import os
from pathlib import Path
import time
import unittest

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "hogwarts-flyover.html"
GPU = os.environ.get("HOG_GPU", "").lower()
GPU_ARGS = {"amd": ["--force-low-power-gpu"], "nvidia": ["--force-high-performance-gpu"]}
RESET_TIMEOUT_MS = 15000            # frames that stop coming for this long mean the page's frame loop has died
START_TIMEOUT_MS = 300000           # loading the page and compiling its programs, two minutes on a busy machine


def gpu_args():
    """Edge's flags for HOG_GPU."""
    if GPU and GPU not in GPU_ARGS:
        raise ValueError(f"HOG_GPU={GPU!r}: use one of {sorted(GPU_ARGS)}, or leave it unset for the browser's default")
    return GPU_ARGS.get(GPU, [])


async def check_adapter(page):
    """The adapter the page renders on; skips as blocked when HOG_GPU names another."""
    adapter = await page.evaluate("document.getElementById('gpu-info').textContent")
    if GPU and GPU not in adapter.lower():
        raise unittest.SkipTest(f"blocked: HOG_GPU={GPU}, but Edge renders on {adapter!r}")
    return adapter


def smoke(test):
    """Tag a test for the smoke subset that lane milestones run on NVIDIA (python -m tools.testrun smoke)."""
    test.smoke = True
    return test


class SharedEdge:
    """The test process's Edge: launched when a class first needs it, again when it has gone, and closed at exit."""

    def __init__(self):
        self.loop = self.playwright = self.browser = None

    def event_loop(self):
        """The loop the classes run on: Playwright's objects work only on the loop that made them."""
        if self.loop is None or self.loop.is_closed():
            self.loop = asyncio.new_event_loop()
        return self.loop

    async def launched(self):
        """The browser, launched afresh unless it is running."""
        if self.browser is None or not self.browser.is_connected():
            await self.stop()
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(channel="msedge", headless=True,
                                                                 chromium_sandbox=True, args=gpu_args())
        return self.browser

    async def stop(self):
        """Close the browser and stop Playwright, whatever state they are in."""
        browser, playwright = self.browser, self.playwright
        self.browser = self.playwright = None
        if browser:
            with contextlib.suppress(Exception):
                await browser.close()
        if playwright:
            with contextlib.suppress(Exception):
                await playwright.stop()

    def close(self):
        """Stop the browser and close the loop, if they are open."""
        if self.loop is None or self.loop.is_closed():
            return
        try:
            self.loop.run_until_complete(self.stop())
        finally:
            self.loop.close()


EDGE = SharedEdge()
atexit.register(EDGE.close)


def _on_class_loop(test):
    @functools.wraps(test)
    def run(self):
        return self.loop.run_until_complete(test(self))
    return run


class EdgeTestCase(unittest.TestCase):
    """Tests whose coroutines run on the event loop of the browser they open their contexts in."""
    edge = EDGE

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, member in list(vars(cls).items()):
            if name.startswith("test") and inspect.iscoroutinefunction(member):
                setattr(cls, name, _on_class_loop(member))

    @classmethod
    def setUpClass(cls):
        cls.loop = cls.edge.event_loop()

    @classmethod
    def _fixture(cls, work):
        """Run fixture `work`; if it fails, close the browser too, so the next fixture launches a fresh one rather than
        inherit what broke."""
        try:
            cls.loop.run_until_complete(work)
        except BaseException:
            cls.loop.run_until_complete(cls.edge.stop())
            raise


class PageTestCase(EdgeTestCase):
    QUERY = "?t=60&pause=1&scale=0.35&hud=0"
    OPENING_T, TOD, DBG = 60, 18.5, 0           # what the page opens with for QUERY (template_c.html's defaults)
    VIEWPORT = {"width": 1100, "height": 720}
    INIT_SCRIPTS = ()                           # scripts the page runs before its own

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        started = time.perf_counter()
        cls._fixture(cls._launch())
        cls.load_s = time.perf_counter() - started

    @classmethod
    def tearDownClass(cls):
        cls._fixture(cls.context.close())

    @classmethod
    async def _launch(cls):
        browser = await cls.edge.launched()
        cls.context = await browser.new_context(viewport=cls.VIEWPORT, offline=True)
        for script in cls.INIT_SCRIPTS:
            await cls.context.add_init_script(script=script)
        await cls._open()

    @classmethod
    async def _open(cls):
        cls.page = await cls.context.new_page()
        cls.errors = []
        cls.page.on("pageerror", lambda error: cls.errors.append(str(error)))
        await cls.page.goto(PAGE.as_uri() + cls.QUERY)
        await cls.page.wait_for_function("window.__hog && (__hog.error || (__hog.ready && __hog.frames >= 10))",
                                         timeout=START_TIMEOUT_MS)
        error = await cls.page.evaluate("__hog.error || null")
        if error:
            raise RuntimeError(f"the page failed to start: {error}")
        cls.adapter = await check_adapter(cls.page)

    async def _alive(self):
        try:
            return not self.page.is_closed() and await self.page.evaluate("!!(window.__hog && __hog.ready && !__hog.error)")
        except Exception:
            return False

    async def _restore(self):
        """What tests change, as the page opened: the viewport, the time of day, the debug bits, meshes on with the
        embedded payload, and the opening view. Times out when the page's frame loop has stopped."""
        cls = type(self)
        if cls.page.viewport_size != self.VIEWPORT:
            await cls.page.set_viewport_size(self.VIEWPORT)
        frames = await cls.page.evaluate("""async ([t, tod, dbg]) => {
            __hog.tod(tod); __hog.dbg(dbg); __hog.meshes(true);
            await __hog.loadMeshes(typeof MESHES === "undefined" ? null : MESHES);
            const p = __hog.path(t);
            __hog.camera(p.pos, Math.atan2(p.fwd[0], p.fwd[2]), Math.asin(p.fwd[1]));
            return __hog.frames; }""", [self.OPENING_T, self.TOD, self.DBG])
        await cls.page.wait_for_function(f"__hog.frames >= {frames + 2}", timeout=RESET_TIMEOUT_MS)

    async def _reset(self):
        cls = type(self)
        if await self._alive():
            try:
                return await self._restore()
            except Exception:                           # the frame loop has stopped, or the page died meanwhile
                pass
        if not cls.page.is_closed():
            await cls.page.close()
        await cls._open()
        await self._restore()

    async def _settle(self):
        """One more frame, so page errors raised by the test's last frames reach `errors` before teardown checks it."""
        with contextlib.suppress(Exception):
            frames = await self.page.evaluate("__hog.frames")
            await self.page.wait_for_function(f"__hog.frames > {frames}", timeout=RESET_TIMEOUT_MS)

    def setUp(self):
        self.loop.run_until_complete(self._reset())

    def tearDown(self):
        self.loop.run_until_complete(self._settle())
        errors = list(self.errors)
        self.errors.clear()
        self.assertEqual(errors, [])
