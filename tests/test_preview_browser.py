"""Optional browser regression. Install Playwright and its browsers, then set TPL_PREVIEW_BROWSERS=chromium,firefox."""
import os
from pathlib import Path
import shutil
import unittest


@unittest.skipUnless(os.getenv("TPL_PREVIEW_BROWSERS"), "opt in to the Playwright preview layout test")
class PreviewLayoutTest(unittest.TestCase):
    def test_compact_viewport(self):
        from playwright.sync_api import sync_playwright

        source = (Path(__file__).resolve().parents[1] / "web/feedback.js").read_text(encoding="utf-8")
        script = "\n".join(line for line in source.splitlines() if not line.startswith("import "))
        with sync_playwright() as playwright:
            for engine in map(str.strip, os.environ["TPL_PREVIEW_BROWSERS"].split(",")):
                with self.subTest(browser=engine):
                    browser_type = getattr(playwright, engine)
                    binary = browser_type.executable_path
                    if not Path(binary).is_file() and engine == "chromium":
                        binary = shutil.which("chromium")
                    browser = browser_type.launch(executable_path=binary)
                    try:
                        self.check_preview(browser.new_page(), script)
                    finally:
                        browser.close()

    def check_preview(self, page, script):
        page.set_content("""
          <div id="host" style="width:351px;height:192px;transform:translate(14px,0)"></div>
          <button id="neighbor">Another node</button>
        """)
        page.evaluate("""() => {
          const host = document.querySelector('#host');
          window.node = { comfyClass: 'CaptionPreviewTpl',
            addDOMWidget(name, type, element) { host.replaceChildren(element); } };
          window.app = { registerExtension(ext) { ext.nodeCreated(node); } };
          window.api = { apiURL: () => 'data:image/svg+xml,' + encodeURIComponent(
            '<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512"><rect width="512" height="512" fill="blue"/></svg>') };
        }""")
        page.add_script_tag(content=script)
        page.evaluate("""() => {
          window.output = { text: ['fallback'], tpl_summary: ['Preview of four images'], tpl_cards:
            Array.from({length: 4}, (_, i) => ({
              filename: i + '.png', image: {filename: i + '.png', label: 'Picture ' + i},
              captions: [{label: 'Detailed Description', text: 'A long caption ' + 'word '.repeat(75) + 'X'.repeat(100)}],
              details: 'Provenance ' + i
            })) };
          node.tplRenderFeedback(output);
        }""")
        frame = page.locator("#host > div")
        page.locator("article img").first.evaluate("img => img.decode()")
        self.assertEqual(page.locator("article").count(), 4)
        # Verify the thumbnail and the first caption line are actually painted inside
        # the transformed viewport. Visible styles or a nonzero card rect are insufficient.
        self.assertTrue(page.evaluate("""() => {
          const card = document.querySelector('article');
          const caption = [...card.querySelectorAll('div')].find(el => el.firstChild?.nodeType === 3
            && el.textContent.startsWith('A long caption'));
          const range = document.createRange();
          range.setStart(caption.firstChild, 0); range.setEnd(caption.firstChild, 1);
          const hit = (rect, target) => target.contains(document.elementFromPoint(
            rect.x + rect.width / 2, rect.y + rect.height / 2));
          const img = card.querySelector('img');
          return hit(img.getBoundingClientRect(), img) && hit(range.getBoundingClientRect(), caption);
        }"""))
        self.assertTrue(frame.evaluate("el => el.clientWidth >= el.scrollWidth"))
        page.locator("article strong").first.click()
        page.mouse.wheel(0, 400)
        page.wait_for_function("document.querySelector('#host > div').scrollTop > 0")
        page.locator("summary").last.click()
        self.assertTrue(page.get_by_text("Provenance 3", exact=True).is_visible())
        self.assertEqual(frame.evaluate("el => el.getBoundingClientRect().height"), 192)
        self.assertTrue(page.locator("#neighbor").evaluate("""el => {
          const r = el.getBoundingClientRect(); return document.elementFromPoint(r.x+10,r.y+4) === el;
        }"""))
        # Vue's grid may give custom DOM widgets intrinsic height instead of a fixed height.
        page.locator("#host").evaluate("el => el.style.height = 'auto'")
        self.assertTrue(frame.evaluate("el => el.clientHeight >= 100 && el.clientHeight < el.scrollHeight"))
