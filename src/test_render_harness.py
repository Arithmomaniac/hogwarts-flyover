"""The render suite's classes share one Edge per test process, each class in its own browser context and page, so only
the process's first class waits for Edge to compile the page's programs. A class whose setup or teardown fails takes the
browser with it, so the next class starts a fresh one rather than inherit what broke."""
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pagetest  # noqa: E402


def blank(edge=None):
    """A PageTestCase class whose page is blank, not the flyover, on `edge` (by default the browser the suite shares)."""
    class Blank(pagetest.PageTestCase):
        @classmethod
        async def _open(cls):
            cls.page = await cls.context.new_page()
    if edge:
        Blank.edge = edge
    return Blank


class SharedBrowserTests(unittest.TestCase):
    def private_edge(self):
        edge = pagetest.SharedEdge()
        self.addCleanup(edge.close)
        return edge

    def test_classes_share_one_browser_each_in_its_own_context(self):
        first, second = blank(), blank()
        for cls in (first, second):
            cls.setUpClass()
            self.addCleanup(cls.tearDownClass)
        browser = first.context.browser
        self.assertIs(second.context.browser, browser)
        self.assertIsNot(second.context, first.context)
        first.tearDownClass()
        self.assertNotIn(first.context, browser.contexts)
        self.assertEqual(second.loop.run_until_complete(second.page.evaluate("6 * 7")), 42)
        second.tearDownClass()
        self.assertTrue(browser.is_connected())         # still running for the next class

    def test_a_browser_that_has_gone_is_launched_again(self):
        edge = self.private_edge()
        loop = edge.event_loop()
        gone = loop.run_until_complete(edge.launched())
        loop.run_until_complete(gone.close())
        browser = loop.run_until_complete(edge.launched())
        self.assertIsNot(browser, gone)
        self.assertTrue(browser.is_connected())

    def test_a_class_that_fails_to_start_takes_the_browser_with_it(self):
        class Failing(pagetest.PageTestCase):
            @classmethod
            async def _open(cls):
                raise RuntimeError("the page failed to start")
        Failing.edge = self.private_edge()
        with self.assertRaises(RuntimeError):
            Failing.setUpClass()
        self.assertFalse(Failing.context.browser.is_connected())

    def test_a_class_whose_context_fails_to_close_takes_the_browser_with_it(self):
        stuck = blank(self.private_edge())
        stuck.setUpClass()
        browser = stuck.context.browser
        stuck.context = mock.Mock(close=mock.AsyncMock(side_effect=RuntimeError("the context did not close")))
        with self.assertRaises(RuntimeError):
            stuck.tearDownClass()
        self.assertFalse(browser.is_connected())
