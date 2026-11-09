"""Annotated screenshots of notebook cells (Colab-like HTML from render.py).

Each shot: a range of cells captured as one image; each mark outlines the smallest element in a cell that
contains a piece of text and puts a numbered red badge next to it.
"""
import asyncio
import json
import sys
from pathlib import Path

from playwright.async_api import async_playwright

CHROME = ("/Users/simone.bonato/Library/Caches/ms-playwright/chromium_headless_shell-1234/"
          "chrome-headless-shell-mac-arm64/chrome-headless-shell")
HERE = Path(__file__).parent

MARK_JS = """([cellIdx, text, label, open]) => {
  const cell = document.querySelectorAll('.jp-Cell')[cellIdx];
  if (open) cell.querySelectorAll('details').forEach(d => d.open = true);
  if (!text) return true;
  const walker = document.createTreeWalker(cell, NodeFilter.SHOW_TEXT);
  let best = null;
  while (walker.nextNode()) {
    if (walker.currentNode.textContent.includes(text)) { best = walker.currentNode.parentElement; break; }
  }
  if (!best) { // text split across nodes: smallest element whose text contains it
    for (const el of cell.querySelectorAll('*')) if (el.textContent.includes(text)) best = el;
  }
  if (!best) return false;
  best.style.outline = '3px solid #D7263D'; best.style.outlineOffset = '2px'; best.style.borderRadius = '3px';
  const r = best.getBoundingClientRect();
  const badge = document.createElement('div');
  badge.textContent = label;
  badge.style.cssText = 'position:absolute;z-index:9999;background:#D7263D;color:white;font:bold 15px sans-serif;' +
    'border-radius:12px;padding:2px 9px;box-shadow:0 1px 3px rgba(0,0,0,.3)';
  badge.style.left = (window.scrollX + Math.max(4, r.left - 40)) + 'px';
  badge.style.top = (window.scrollY + r.top - 4) + 'px';
  document.body.appendChild(badge);
  return true;
}"""


async def main(spec_path: str) -> None:
    specs = json.loads(Path(spec_path).read_text())
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=CHROME)
        pages = {}
        for s in specs:
            html = s["html"]
            if html not in pages:
                pg = await browser.new_page(viewport={"width": 1100, "height": 1400}, device_scale_factor=2)
                await pg.goto("file://" + str(HERE / html))
                await pg.add_style_tag(content="body{background:white;padding-left:44px} .jp-InputArea-editor{max-height:none}")
                await pg.wait_for_timeout(800)
                pages[html] = pg
            pg = pages[html]
            first, last = s["cells"]
            for m in s.get("marks", []):
                ok = await pg.evaluate(MARK_JS, [m["cell"], m.get("text"), m.get("label", ""), m.get("open", False)])
                if not ok:
                    print("mark not found:", s["out"], m)
            cells = pg.locator(".jp-Cell")
            boxes = [await cells.nth(i).bounding_box() for i in range(first, last + 1)]
            boxes = [b for b in boxes if b]
            top = min(b["y"] for b in boxes) - 8
            bottom = max(b["y"] + b["height"] for b in boxes) + 8
            await pg.screenshot(path=str(HERE / "img" / s["out"]), full_page=True,
                                clip={"x": 0, "y": max(0, top), "width": 1144,
                                      "height": min(bottom - top, s.get("max_h", 1600))})
            print("shot", s["out"])
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
