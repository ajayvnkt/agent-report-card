"""Capture README screenshots into docs/images (app must run on http://localhost:9003)."""
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs/images"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5, color_scheme="light")
    pg.goto("http://localhost:9003"); pg.get_by_text("Step 1").first.wait_for(timeout=60000); pg.wait_for_timeout(2500)
    pg.screenshot(path=IMG / "01_steps.png")
    pg.locator('input[type="file"]').first.set_input_files(str(ROOT / "sample_logs/demo_agent_runs.log"))
    pg.get_by_text("Detected:").first.wait_for(timeout=30000); pg.wait_for_timeout(1500)
    pg.get_by_text("SkyDesk policy", exact=True).first.click(); pg.wait_for_timeout(1500)
    exp = pg.locator("summary").filter(has_text="What should a policy look like?").first
    exp.click(); pg.wait_for_timeout(1500); exp.scroll_into_view_if_needed(); pg.mouse.wheel(0, -60); pg.wait_for_timeout(800)
    pg.screenshot(path=IMG / "02_policy_format.png"); exp.click(); pg.wait_for_timeout(800)
    pg.get_by_role("button", name="▶ Run analysis").click()
    pg.get_by_text("Analysis complete").first.wait_for(timeout=60000); pg.wait_for_timeout(2000)
    for i, (tab, anchor) in enumerate([("Summary", "Your report card"), ("Action items", None),
                                       ("Run detail", None), ("Model accuracy", None)], start=3):
        pg.get_by_role("tab", name=tab).first.click(); pg.wait_for_timeout(2000)
        pg.get_by_role("tab", name=tab).first.evaluate("e => e.scrollIntoView({block: 'start'})"); pg.mouse.wheel(0, -20); pg.wait_for_timeout(900)
        pg.screenshot(path=IMG / f"0{i}_{tab.lower().replace(' ', '_')}.png")
    b.close()
print(sorted(x.name for x in IMG.iterdir()))
