#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "presentation.html").resolve()
OUT = ROOT / "presentation_assets" / "screenshots"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    url = HTML.as_uri()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=1)
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(1500)
        total = page.evaluate("Reveal.getTotalSlides()")
        for i in range(total):
            page.evaluate(f"Reveal.slide({i})")
            page.wait_for_timeout(800)
            page.locator(".reveal").screenshot(path=str(OUT / f"slide_{i+1:02d}.png"))
        browser.close()
    print(f"wrote {total} screenshots to {OUT}")


if __name__ == "__main__":
    main()
