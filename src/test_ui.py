"""Browser checks for the standalone page's navigation and scene controls."""
import re
import unittest
from pathlib import Path

from playwright.async_api import async_playwright, expect


class OverlayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            channel="msedge", headless=True, chromium_sandbox=True,
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
        await self.page.wait_for_function(
            "window.__hog && __hog.ready && getComputedStyle(document.getElementById('load')).opacity === '0'",
            timeout=120000,
        )

    async def asyncTearDown(self):
        await self.browser.close()
        await self.playwright.stop()
        self.assertEqual(self.errors, [])

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

    async def test_frame_stats_use_wall_time_without_changing_movement_timing(self):
        page = self.page
        await page.get_by_text("Stats", exact=True).click()
        await page.evaluate("""async () => {
          const native = window.requestAnimationFrame.bind(window);
          const clock = window.__testClock = {native, next:null, now:0};
          window.requestAnimationFrame = callback => {
            if (callback.name !== "loop") return native(callback);
            clock.next = callback;
            return 0;
          };
          await new Promise(resolve => native(now => { clock.now = now; resolve(); }));
          clock.step = ms => { clock.now += ms; clock.next(clock.now); };
        }""")
        try:
            await page.evaluate("() => { for(let i=0;i<120;i++) __testClock.step(250); }")
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
        finally:
            await page.evaluate("""() => {
              const clock=__testClock;
              window.requestAnimationFrame=clock.native;
              clock.native(clock.next);
              delete window.__testClock;
            }""")

    async def test_gpu_details_unavailable_does_not_prevent_rendering(self):
        page = self.page
        await page.add_init_script("""(() => {
          const getExtension=WebGL2RenderingContext.prototype.getExtension;
          WebGL2RenderingContext.prototype.getExtension=function(name) {
            return name==="WEBGL_debug_renderer_info" ? null : getExtension.call(this,name);
          };
        })()""")
        await page.reload()
        await page.wait_for_function("window.__hog && __hog.ready", timeout=120000)
        await page.wait_for_function(
            "getComputedStyle(document.getElementById('load')).opacity === '0'",
        )
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
