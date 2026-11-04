"""Browser checks for the standalone page's navigation and scene controls. HOG_GPU pins the adapter (see pagetest)."""
import contextlib
import re
import sys
import unittest
from pathlib import Path

from playwright.async_api import async_playwright, expect

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pagetest import check_adapter, gpu_args  # noqa: E402

INSTALL_STEPPED_CLOCK = """async () => {
  const native = window.requestAnimationFrame.bind(window);
  const clock = window.__testClock = {native, next:null, now:0};
  window.requestAnimationFrame = callback => {
    if (callback.name !== "loop") return native(callback);
    clock.next = callback;
    return 0;
  };
  await new Promise(resolve => native(now => { clock.now = now; resolve(); }));
  clock.step = ms => { clock.now += ms; clock.next(clock.now); };
}"""

RESTORE_NATIVE_CLOCK = """() => {
  const clock=__testClock;
  window.requestAnimationFrame=clock.native;
  clock.native(clock.next);
  delete window.__testClock;
}"""


class OverlayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            channel="msedge", headless=True, chromium_sandbox=True, args=gpu_args(),
        )
        self.context = await self.browser.new_context(
            viewport={"width": 1100, "height": 720}, offline=True,
        )
        self.page = await self.context.new_page()
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.set_default_timeout(3000)
        url = (Path(__file__).resolve().parents[1] / "hogwarts-flyover.html").as_uri()
        await self.page.goto(url + "?t=60&pause=1&scale=0.35")
        await self.wait_until_interactive()
        try:
            await check_adapter(self.page)
        except unittest.SkipTest:
            await self.browser.close()
            await self.playwright.stop()
            raise

    async def wait_until_interactive(self):
        # the first draws after "ready" can stall for several seconds while the driver compiles shader variants
        await self.page.wait_for_function(
            "window.__hog && __hog.ready && __hog.frames >= 30"
            " && getComputedStyle(document.getElementById('load')).opacity === '0'",
            timeout=120000,
        )

    async def asyncTearDown(self):
        await self.browser.close()
        await self.playwright.stop()
        self.assertEqual(self.errors, [])

    @contextlib.asynccontextmanager
    async def stepped_loop_clock(self):
        """Drive the render loop with __testClock.step(ms) instead of the browser's frame clock."""
        await self.page.evaluate(INSTALL_STEPPED_CLOCK)
        try:
            yield
        finally:
            await self.page.evaluate(RESTORE_NATIVE_CLOCK)

    async def test_dialogs_isolate_scene_keys_and_restore_focus_and_playback(self):
        page = self.page
        status = page.locator("#status")
        about = page.get_by_role("button", name="About", exact=True)
        await about.click()
        dialog = page.get_by_role("dialog", name="About this flyover")
        await expect(dialog).to_be_visible()
        await expect(status).to_contain_text("flyover 60")
        await dialog.focus()
        await page.keyboard.press("f")
        await page.keyboard.press("ArrowRight")
        await page.keyboard.press("Space")
        await page.keyboard.press("h")
        await expect(status).to_contain_text("flyover 60")
        await expect(page.locator("#panel")).to_be_visible()
        await page.keyboard.press("Tab")
        close = dialog.get_by_role("button", name="Close")
        await expect(close).to_be_focused()
        await page.keyboard.press("Shift+Tab")
        await expect(dialog.get_by_role("link", name="Rebuild and reconstruction guide")).to_be_focused()
        await page.keyboard.press("Tab")
        await expect(close).to_be_focused()
        await page.keyboard.press("Escape")
        await expect(dialog).not_to_be_visible()
        await expect(about).to_be_focused()
        await expect(page.get_by_role("button", name="Resume tour")).to_be_visible()

        await page.get_by_role("button", name="Resume tour").click()
        await expect(page.get_by_role("button", name="Pause tour")).to_be_visible()
        await about.click()
        await expect(dialog).to_be_visible()
        await expect(page.locator("#play")).to_have_text("Resume tour")
        await dialog.get_by_role("button", name="Close").click()
        await expect(page.get_by_role("button", name="Pause tour")).to_be_visible()

        await page.get_by_role("button", name="Controls", exact=True).click()
        await expect(page.get_by_role("dialog", name="Controls")).to_be_visible()
        await page.keyboard.press("Space")
        await expect(page.get_by_role("dialog", name="Controls")).not_to_be_visible()
        await page.get_by_role("button", name="Credits", exact=True).click()
        credits = page.get_by_role("dialog", name="Sources and credits")
        await expect(credits).to_be_visible()
        await expect(credits).to_contain_text("CC BY-NC-SA 4.0")

    async def test_playback_stats_and_native_keyboard_controls(self):
        page = self.page
        status = page.locator("#status")
        await expect(status).to_contain_text("flyover 60")
        await page.get_by_role("button", name="Forward 10 seconds").click()
        await expect(status).to_contain_text("flyover 70")
        await page.get_by_role("button", name="Back 10 seconds").click()
        await expect(status).to_contain_text("flyover 60")
        await expect(page.locator("#stats")).not_to_be_visible()
        await page.get_by_text("Stats", exact=True).click()
        await expect(page.locator("#stats")).to_be_visible()
        await expect(page.locator("#stats")).to_contain_text("above ground")
        renderer = await page.locator("#c").evaluate(
            "c => { const g=c.getContext('webgl2'), e=g.getExtension('WEBGL_debug_renderer_info');"
            "return e && g.getParameter(e.UNMASKED_RENDERER_WEBGL); }"
        )
        self.assertTrue(renderer)
        await expect(page.locator("#gpu-info")).to_have_text("GPU: " + renderer)
        slider = page.get_by_role("slider", name="Time of day")
        before = float(await slider.input_value())
        await slider.focus()
        await page.keyboard.press("ArrowRight")
        self.assertGreater(float(await slider.input_value()), before)
        await expect(status).to_contain_text("flyover 60")
        await page.locator("#c").focus()
        await page.keyboard.press("ArrowRight")
        await expect(status).to_contain_text("flyover 70")
        await page.keyboard.press("h")
        await expect(page.locator("#panel")).not_to_be_visible()
        await page.keyboard.press("h")
        await expect(page.locator("#panel")).to_be_visible()

    async def test_the_status_shows_the_whole_second_the_tour_is_at(self):
        # wrapping the tour time as ((t % total) + total) % total rounds some whole seconds down to the one before, and
        # which ones depends on the tour's length, so pick one that this page's length rounds down
        page = self.page
        second = await page.evaluate("""() => { const T = __hog.total;
            for (let k = 1; k < T; k++) if (((k % T) + T) % T < k) return k;
            return null; }""")
        if second is None:
            self.skipTest("this tour's length rounds no whole second down")
        await page.evaluate("(k) => __hog.set(k)", second)
        await expect(page.locator("#status")).to_contain_text(f"flyover {second} /")

    async def test_frame_stats_use_wall_time_without_changing_movement_timing(self):
        page = self.page
        await page.get_by_text("Stats", exact=True).click()
        async with self.stepped_loop_clock():
            # a converged still view stops rendering and sampling frame gaps, so keep the tour moving
            await page.evaluate("() => { __hog.pause(false); for(let i=0;i<120;i++) __testClock.step(250); }")
            stats = page.locator("#stats")
            await expect(stats).to_contain_text("4 loop fps")
            await expect(stats).to_contain_text("median 250.0 ms")
            await expect(stats).to_contain_text("p95 250.0 ms")
            await expect(stats).to_contain_text("max 250.0 ms")
            await page.evaluate("() => { __testClock.step(1000); for(let i=0;i<9;i++) __testClock.step(250); }")
            await expect(stats).to_contain_text("p95 250.0 ms")
            await expect(stats).to_contain_text("max 1000.0 ms")
            await page.evaluate("() => { for(let i=0;i<130;i++) __testClock.step(250); }")
            await expect(stats).to_contain_text("max 250.0 ms")
            await page.evaluate("""() => {
              __hog.set(60); __hog.pause(false);
              for(let i=0;i<20;i++) __testClock.step(250);
            }""")
            await expect(page.locator("#status")).to_contain_text(re.compile(r"flyover 6[12] /"))

    async def test_converged_still_view_skips_rendering_until_scene_changes(self):
        page = self.page
        status = page.locator("#status")
        await page.get_by_text("Stats", exact=True).click()
        async with self.stepped_loop_clock():
            await page.evaluate("""() => {
              const proto = WebGL2RenderingContext.prototype, drawArrays = proto.drawArrays, drawElements = proto.drawElements;
              window.__draws = 0;
              window.__meshDraws = 0;
              proto.drawArrays = function(...args) {
                __draws++;
                return drawArrays.apply(this, args);
              };
              proto.drawElements = function(...args) {
                __meshDraws++;
                return drawElements.apply(this, args);
              };
              for(let i=0;i<120;i++) __testClock.step(250);
            }""")
            await expect(status).to_contain_text("still")
            still = await page.evaluate("""() => {
              const draws = __draws, meshDraws = __meshDraws, frames = __hog.frames;
              __testClock.step(1000);
              for(let i=0;i<19;i++) __testClock.step(250);
              return {draws: __draws - draws, meshDraws: __meshDraws - meshDraws, frames: __hog.frames - frames};
            }""")
            self.assertEqual(still, {"draws": 0, "meshDraws": 0, "frames": 20})
            await expect(page.locator("#stats")).not_to_contain_text("max 1000.0 ms")

            slider = page.get_by_role("slider", name="Time of day")
            await slider.focus()
            await page.keyboard.press("ArrowRight")
            resumed = await page.evaluate("""() => {
              const draws = __draws, meshDraws = __meshDraws;
              for(let i=0;i<10;i++) __testClock.step(250);
              return [__draws - draws, __meshDraws - meshDraws];
            }""")
            self.assertGreater(min(resumed), 0)
            await expect(status).not_to_contain_text("still")

    async def test_gpu_details_unavailable_does_not_prevent_rendering(self):
        page = self.page
        await page.add_init_script("""(() => {
          const getExtension=WebGL2RenderingContext.prototype.getExtension;
          WebGL2RenderingContext.prototype.getExtension=function(name) {
            return name==="WEBGL_debug_renderer_info" ? null : getExtension.call(this,name);
          };
        })()""")
        await page.reload()
        await self.wait_until_interactive()
        await page.get_by_text("Stats", exact=True).click()
        await expect(page.locator("#gpu-info")).to_have_text("GPU: adapter details unavailable")
        await expect(page.locator("#stats")).to_contain_text("loop fps")

    async def test_mobile_layout_and_scrollable_dialog(self):
        page = self.page
        viewport = {"width": 390, "height": 844}
        await page.set_viewport_size(viewport)
        for selector in ("#title", "#panel", "#playback", "#credits", "#status"):
            with self.subTest(selector=selector):
                box = await page.locator(selector).bounding_box()
                self.assertGreaterEqual(box["x"], 0)
                self.assertLessEqual(box["x"] + box["width"], viewport["width"])
                self.assertLessEqual(box["y"] + box["height"], viewport["height"])
        await page.get_by_text("Stats", exact=True).click()
        box = await page.locator("#stats").bounding_box()
        self.assertGreaterEqual(box["x"], 0)
        self.assertLessEqual(box["x"] + box["width"], viewport["width"])
        self.assertLessEqual(await page.locator("#stats").evaluate("s => s.scrollWidth"), box["width"])
        await page.get_by_role("button", name="Controls", exact=True).click()
        dialog = page.get_by_role("dialog", name="Controls")
        await expect(dialog).to_be_visible()
        box = await dialog.bounding_box()
        self.assertGreaterEqual(box["y"], 0)
        self.assertLessEqual(box["y"] + box["height"], viewport["height"])
        self.assertLessEqual(
            await dialog.evaluate("d => d.scrollWidth"), box["width"],
        )
        await page.keyboard.press("Escape")
        self.assertTrue(await page.locator("#c").evaluate(
            "c => { const g=c.getContext('webgl2'), e=g.getExtension('WEBGL_debug_renderer_info');"
            "return e && !/swiftshader|llvmpipe|software|warp/i.test(g.getParameter(e.UNMASKED_RENDERER_WEBGL)); }"
        ))


if __name__ == "__main__":
    unittest.main()
