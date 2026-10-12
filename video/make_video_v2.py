"""Record the click-by-click demo of Agent Report Card with a natural voice (Kokoro, am_michael).

App must be running on http://localhost:9003. The script is data-driven: video/narration.json is a list of scenes,
each a list of beats {"key", "say", "do"}. For every beat the audio is synthesised on its own, the beat's action
(ACTIONS[key]) runs the moment its words start, and the screen then holds only for the rest of that beat's audio,
with gentle cursor motion over the thing being described. The opening (the flagship run in Run detail) is prepared
silently before the recorded segment starts and trimmed away.

    video/.venv/bin/python video/make_video_v2.py [--dry] [--voice am_michael] [--speed 1.0]

Outputs video/out/agent_report_card_demo_v2.mp4 (narrated), ..._v2_silent.mp4, ..._v2.srt, beat_timeline.json
and one frame per beat in video/out/beat_frames/ (taken right after the beat's action) for checking.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
ROOT = HERE.parent
OUT = HERE / "out"
TTS_CACHE = OUT / "tts"
FRAMES = OUT / "beat_frames"
URL = "http://localhost:9003"
DEMO_LOG = ROOT / "sample_logs/demo_agent_runs.log"
W, H = 1440, 900
SR = 24000            # Kokoro sample rate
GAP = 0.25            # silence between beats (s)
MAX_BEAT = 9.5        # a beat's speech must stay under this (s)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# A visible cursor + click ripple (headless recordings have no pointer) and a highlight box that follows an element.
OVERLAY_JS = """
(() => {
  const add = () => {
    if (document.getElementById('__demo_cursor')) return;
    const c = document.createElement('div');
    c.id = '__demo_cursor';
    Object.assign(c.style, {position:'fixed', left:'-50px', top:'-50px', width:'22px', height:'22px',
      borderRadius:'50%', background:'rgba(255,75,75,0.85)', border:'3px solid white',
      boxShadow:'0 0 0 2px rgba(255,75,75,0.6), 0 2px 8px rgba(0,0,0,.35)', zIndex:2147483647,
      pointerEvents:'none', transform:'translate(-50%,-50%)', transition:'transform 120ms'});
    document.body.appendChild(c);
    document.addEventListener('mousemove', e => { c.style.left = e.clientX+'px'; c.style.top = e.clientY+'px'; }, true);
    document.addEventListener('mousedown', e => {
      c.style.transform = 'translate(-50%,-50%) scale(0.7)';
      const r = document.createElement('div');
      Object.assign(r.style, {position:'fixed', left:e.clientX+'px', top:e.clientY+'px', width:'14px', height:'14px',
        borderRadius:'50%', border:'3px solid rgba(255,75,75,0.9)', transform:'translate(-50%,-50%)',
        zIndex:2147483646, pointerEvents:'none', transition:'all 450ms ease-out', opacity:'1'});
      document.body.appendChild(r);
      requestAnimationFrame(() => { r.style.width='60px'; r.style.height='60px'; r.style.opacity='0'; });
      setTimeout(() => r.remove(), 500);
    }, true);
    document.addEventListener('mouseup', () => { c.style.transform = 'translate(-50%,-50%)'; }, true);
    const box = document.createElement('div');
    box.id = '__demo_hl';
    Object.assign(box.style, {position:'fixed', border:'3px solid rgba(255,75,75,0.95)', borderRadius:'10px',
      background:'rgba(255,75,75,0.06)', boxShadow:'0 0 0 4px rgba(255,75,75,0.15)', zIndex:2147483645,
      pointerEvents:'none', opacity:'0', transition:'opacity 250ms'});
    document.body.appendChild(box);
    window.__hlTarget = null;
    window.__hl = (el, pad) => { window.__hlTarget = el; window.__hlPad = pad || 6; box.style.opacity = el ? '1' : '0'; };
    const tick = () => {
      const t = window.__hlTarget;
      if (t && t.isConnected) {
        const r = t.getBoundingClientRect(), p = window.__hlPad;
        Object.assign(box.style, {left:(r.left-p)+'px', top:(r.top-p)+'px', width:(r.width+2*p)+'px', height:(r.height+2*p)+'px'});
        if (!r.width) box.style.opacity = '0';
      } else if (t) { box.style.opacity = '0'; }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };
  if (document.body) add(); else document.addEventListener('DOMContentLoaded', add);
})();
"""


# ----------------------------------------------------------------------------------------------- audio
def synth(beats: list[dict], voice: str, speed: float) -> dict:
    """Per-beat TTS, cached by (voice, speed, text). Returns {key: samples}."""
    TTS_CACHE.mkdir(parents=True, exist_ok=True)
    kokoro = None
    audio = {}
    for b in beats:
        h = hashlib.sha1(f"{voice}|{speed}|{b['say']}".encode()).hexdigest()[:12]
        path = TTS_CACHE / f"{b['key']}_{h}.wav"
        if not path.exists():
            if kokoro is None:
                from kokoro_onnx import Kokoro
                kokoro = Kokoro(str(HERE / "models/kokoro-v1.0.onnx"), str(HERE / "models/voices-v1.0.bin"))
            samples, sr = kokoro.create(b["say"], voice=voice, speed=speed, lang="en-us")
            assert sr == SR, sr
            sf.write(path, samples, sr)
        samples, _ = sf.read(path, dtype="float32")
        audio[b["key"]] = samples
        print(f"tts {b['key']:<18} {len(samples) / SR:4.1f}s")
    return audio


# ----------------------------------------------------------------------------------------------- recorder
class Demo:
    def __init__(self, page, dry: bool):
        self.page, self.dry = page, dry
        self.mx, self.my = W * 0.55, H * 0.45
        self.failures: list[str] = []

    # -- primitives ---------------------------------------------------------------------------------
    def wait(self, ms: int) -> None:
        self.page.wait_for_timeout(ms)

    def vis(self, loc):
        return loc.filter(visible=True).first

    def text(self, s: str, exact: bool = False):
        return self.vis(self.page.get_by_text(s, exact=exact))

    def move(self, x: float, y: float, steps: int = 16) -> None:
        self.page.mouse.move(x, y, steps=steps)
        self.mx, self.my = x, y

    def scroll_to(self, loc, block: str = "center") -> None:
        loc.wait_for(state="visible", timeout=10000)
        loc.evaluate(f"e => {{ e.style.scrollMarginTop = '90px'; e.scrollIntoView({{behavior: 'smooth', block: '{block}'}}); }}")
        self.wait(700)

    def point(self, loc, steps: int = 16, dx: float = 0.5) -> None:
        b = loc.bounding_box()
        if b:
            self.move(b["x"] + min(b["width"] * dx, 260), b["y"] + b["height"] / 2, steps)

    def hl(self, loc, pad: int = 6) -> None:
        loc.evaluate(f"e => window.__hl && window.__hl(e, {pad})")

    def unhl(self) -> None:
        self.page.evaluate("() => window.__hl && window.__hl(null)")

    def focus(self, loc, block: str = "center", pad: int = 6, scroll: bool = True) -> None:
        """Glide to an element, point at it and draw the highlight box around it."""
        if scroll:
            self.scroll_to(loc, block)
        self.hl(loc, pad)
        self.point(loc)

    def click(self, loc, after: int = 900, block: str = "center") -> None:
        self.scroll_to(loc, block)
        self.point(loc, steps=14)
        self.wait(150)
        loc.click(timeout=8000)
        self.wait(after)

    def tab(self, name: str) -> None:
        self.unhl()
        self.click(self.vis(self.page.get_by_role("tab", name=name, exact=True)), after=1300)

    def expander(self, label: str):
        return self.vis(self.page.locator("summary").filter(has_text=label))

    def tour(self, locs: list, spread_ms: int, hl: bool = True) -> None:
        """Visit several elements one after another, spread over spread_ms (the words describing them)."""
        each = max(600, spread_ms // max(1, len(locs)))
        for loc in locs:
            t = time.monotonic()
            loc.evaluate("e => { const r = e.getBoundingClientRect();"
                         " if (r.top < 70 || r.bottom > innerHeight - 20) e.scrollIntoView({behavior:'smooth', block:'center'}); }")
            self.wait(350)
            if hl:
                self.hl(loc)
            self.point(loc, steps=20)
            rest = each - int((time.monotonic() - t) * 1000)
            if rest > 1200:
                self.drift(rest)
            elif rest > 0:
                self.wait(rest)

    def sweep(self, lines: int, ms: int, dy: float = 22.0) -> None:
        """Move the cursor down over the next few lines (e.g. of a code block) while they are described."""
        each = max(300, ms // max(1, lines))
        for _ in range(lines):
            self.move(self.mx + 30, self.my + dy, steps=12)
            self.wait(each)

    def drift(self, ms: int) -> None:
        """Gentle motion during a hold: a slow small loop around the current point."""
        t_end = time.monotonic() + ms / 1000
        cx, cy, k = self.mx, self.my, 0
        while time.monotonic() < t_end - 0.45:
            k += 1
            a = k * 1.1
            self.page.mouse.move(cx + 30 * math.cos(a), cy + 12 * math.sin(a), steps=12)
            self.wait(420)
        self.page.mouse.move(cx, cy, steps=6)
        left = t_end - time.monotonic()
        if left > 0:
            self.wait(int(left * 1000))

    # -- app flows ----------------------------------------------------------------------------------
    def fresh(self) -> None:
        self.page.goto(URL)
        self.page.get_by_text("Step 1").first.wait_for(timeout=60000)
        self.wait(1500)

    def upload_demo(self) -> None:
        self.unhl()
        up = self.vis(self.page.get_by_role("button", name="Upload"))
        self.scroll_to(up)
        self.point(up)
        self.wait(150)
        with self.page.expect_file_chooser(timeout=8000) as fc:
            up.click()
        self.wait(300)
        fc.value.set_files(str(DEMO_LOG))
        self.page.get_by_text("Detected:").first.wait_for(timeout=30000)
        self.wait(400)

    def choose_skydesk(self) -> None:
        self.click(self.text("SkyDesk policy", exact=True), after=1200)

    def run_analysis(self) -> None:
        self.click(self.vis(self.page.get_by_role("button", name="▶ Run analysis")), after=200)
        self.page.get_by_text("Analysis complete").first.wait_for(timeout=60000)
        self.wait(800)

    def choose_run(self, rid: str) -> None:
        sb = self.vis(self.page.locator("[data-testid='stSelectbox']").filter(has_text="Choose a run to inspect"))
        self.scroll_to(sb, "start")
        self.point(sb, dx=0.2)
        sb.click()
        self.wait(400)
        self.page.keyboard.type(rid[4:], delay=90)
        self.wait(300)
        self.page.keyboard.press("Enter")
        self.wait(1000)
        self.page.evaluate("() => document.activeElement && document.activeElement.blur()")
        self.wait(300)
        self.scroll_to(sb, "start")

    def claims_box(self):
        return self.page.locator("[data-testid='stVerticalBlock']").filter(
            has=self.page.locator("h5").filter(has_text="Agent CLAIMED")).filter(has_not=self.page.locator("h5").filter(has_text="Logs SHOW")).filter(
            visible=True).last

    def logs_box(self):
        return self.page.locator("[data-testid='stVerticalBlock']").filter(
            has=self.page.locator("h5").filter(has_text="Logs SHOW")).filter(has_not=self.page.locator("h5").filter(has_text="Agent CLAIMED")).filter(
            visible=True).last

    def gauge(self):
        return self.vis(self.page.locator("[data-testid='stPlotlyChart']"))

    def li(self, s: str):
        return self.vis(self.page.locator("li").filter(has_text=s))

    def metric(self, label: str):
        return self.vis(self.page.locator("[data-testid='stMetric']").filter(has_text=label))

    def preroll(self) -> None:
        """Silently prepare the opening: flagship run open in Run detail, claim + logs in view."""
        self.fresh()
        self.upload_demo()
        self.choose_skydesk()
        self.run_analysis()
        self.page.get_by_role("tab", name="Run detail", exact=True).filter(visible=True).first.click()
        self.wait(1500)
        sb = self.vis(self.page.locator("[data-testid='stSelectbox']").filter(has_text="Choose a run to inspect"))
        sb.evaluate("e => e.scrollIntoView({block: 'start'})")
        self.wait(800)
        self.move(W * 0.5, H * 0.3, steps=1)


def show_overlay(page, img_path, caption: str, ms: int) -> None:
    """Full-screen screenshot of the build session, slowly zooming, on top of the app (app state is kept)."""
    import base64
    data = base64.b64encode(Path(img_path).read_bytes()).decode()
    page.evaluate("""([src, cap, ms]) => {
      let o = document.getElementById('__build_overlay');
      if (!o) {
        o = document.createElement('div'); o.id = '__build_overlay';
        Object.assign(o.style, {position:'fixed', inset:'0', background:'#1f2128', zIndex:2147483640,
          display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center',
          opacity:'0', transition:'opacity 350ms'});
        o.innerHTML = '<div id="__bo_cap" style="color:#fff;font:600 26px system-ui;margin:0 0 18px"></div>' +
          '<div style="overflow:hidden;border-radius:10px;box-shadow:0 10px 40px rgba(0,0,0,.5);max-width:86vw;max-height:78vh">' +
          '<img id="__bo_img" style="display:block;max-width:86vw;max-height:78vh;transform-origin:50% 30%"></div>';
        document.body.appendChild(o);
      }
      const img = o.querySelector('#__bo_img');
      img.style.transition = 'none'; img.style.transform = 'scale(1)'; img.src = src;
      o.querySelector('#__bo_cap').textContent = cap;
      requestAnimationFrame(() => { o.style.opacity = '1';
        requestAnimationFrame(() => { img.style.transition = `transform ${ms}ms linear`; img.style.transform = 'scale(1.12)'; }); });
    }""", ["data:image/png;base64," + data, caption, max(ms, 1000)])
    page.wait_for_timeout(400)


def hide_overlay(page) -> None:
    page.evaluate("() => { const o = document.getElementById('__build_overlay'); if (o) { o.style.opacity = '0'; setTimeout(() => o.remove(), 400); } }")
    page.wait_for_timeout(450)


def show_card(page, title: str, sub: str) -> None:
    page.evaluate("""([t, s]) => {
      const c = document.createElement('div'); c.id = '__end_card';
      Object.assign(c.style, {position:'fixed', inset:'0', background:'rgba(255,255,255,.94)', zIndex:2147483640,
        display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', opacity:'0',
        transition:'opacity 500ms', font:'600 44px system-ui', color:'#1f2128'});
      c.innerHTML = '<div>🧾 ' + t + '</div><div style="font:400 22px system-ui;margin-top:16px;color:#555">' + s + '</div>';
      document.body.appendChild(c); requestAnimationFrame(() => c.style.opacity = '1');
    }""", [title, sub])
    page.wait_for_timeout(600)


def build_actions(d: Demo) -> dict:
    p = d.page

    def hook_claim(ms):
        q = d.vis(d.claims_box().locator(".arc-quote"))
        d.focus(q, scroll=False)
        d.drift(ms - 900)

    def hook_logs(ms):
        d.hl(d.logs_box())
        errs = d.logs_box().locator("li").filter(has_text="update_contact")
        d.tour([errs.nth(0), errs.nth(1), errs.nth(2)], ms - 400, hl=False)

    def hook_gap(ms):
        d.focus(d.vis(d.logs_box().get_by_text("Claimed but never succeeded")), scroll=False)
        d.drift(int(ms * 0.55))
        d.focus(d.gauge(), scroll=False)

    def title(ms=0):
        d.unhl()
        h1 = d.vis(p.locator("h1"))
        d.scroll_to(h1, "start")
        d.point(h1, dx=0.6)

    def hook_promise(ms):
        d.unhl()
        d.fresh()
        title()

    def step1_intro(ms):
        d.focus(d.text("Step 1 · Upload your agent logs"))

    def format_open(ms):
        d.unhl()
        d.click(d.expander("What format should my logs be in?"), after=900)
        d.focus(d.text("Short answer: one JSON object per line"), block="start")

    def format_minimal(ms):
        code = d.vis(p.locator("[data-testid='stCode']"))
        d.scroll_to(code)
        d.hl(code)
        lines = [code.get_by_text(t).first for t in ('"user_input"', '"tool_call"', '"tool_result"', '"final_answer"')]
        d.tour(lines, ms - 1200, hl=False)

    def format_accepted(ms):
        d.unhl()
        d.click(d.vis(p.get_by_role("tab", name="Accepted formats")), after=700)
        rows = [d.text(t, exact=True) for t in ("OpenAI chat messages (JSON / JSONL)", "Python logging text (.log / .txt)",
                                                 "OpenTelemetry spans (OTLP JSON or flat spans)", "CSV with ARL column names")]
        d.tour(rows, ms - 1500)

    def format_langfuse(ms):
        d.focus(d.text("Langfuse export (traces + observations, JSON / JSONL)", exact=True))

    def upload(ms):
        d.unhl()
        d.click(d.expander("What format should my logs be in?"), after=500, block="start")
        d.upload_demo()

    def detected(ms):
        det = d.text("Detected:")
        d.focus(det, pad=10)
        d.drift(int(ms * 0.55))
        b = det.bounding_box()
        if b:
            d.move(b["x"] + b["width"] - 60, b["y"] + b["height"] / 2, steps=30)

    def policy_choose(ms):
        d.unhl()
        d.focus(d.text("Step 2 · Policy (optional)"), block="start")
        d.drift(int(ms * 0.35))
        d.unhl()
        d.click(d.text("SkyDesk policy", exact=True), after=900, block="start")

    def policy_plain(ms):
        items = [d.li("issue_refund may only run after"), d.li("never more than lookup_booking"),
                 d.li("retried at most 1")]
        d.tour(items, ms - 300)

    def policy_open(ms):
        d.unhl()
        d.click(d.expander("What should a policy look like?"), after=900)
        d.focus(d.text("Short answer: a small YAML file"), block="start")

    def yaml_key(key, lines):
        def act(ms):
            d.focus(d.vis(p.locator("[data-testid='stCode']").get_by_text(key, exact=True)), block="start", pad=8)
            d.wait(600)
            d.sweep(lines, ms - 2600)
        return act

    def run_click(ms):
        d.unhl()
        d.click(d.expander("What should a policy look like?"), after=400, block="start")
        d.run_analysis()

    def run_done(ms):
        d.focus(d.text("Analysis complete"), pad=8)

    def sum_predicted(ms):
        m = d.metric("Predicted success")
        d.focus(m, block="center")
        d.drift(int(ms * 0.45))
        d.point(d.vis(m.get_by_text("actual 31%")))

    def sum_claimed(ms):
        d.focus(d.metric("Claimed success but likely failed"), scroll=False)

    def sum_critical(ms):
        d.tour([d.metric("Critical policy violations"), d.metric("Est. cost per likely-successful run")], ms - 300)

    def sum_goto(ms):
        d.unhl()
        d.click(d.vis(p.get_by_role("button", name="See all action items")), after=1300)
        d.scroll_to(d.text("11 things to fix"), "start")
        d.point(d.text("11 things to fix"))

    def act_top(ms):
        title = d.text("Agent reports actions that never happened")
        d.scroll_to(d.text("11 things to fix"), "start")
        d.hl(title, 8)
        d.point(title)
        d.drift(ms - 1200)

    def act_fix(ms):
        d.tour([d.text("4 of 16 runs (25%) listed actions"), d.text("Build the user-facing summary from the audit log")],
               ms - 300)

    def act_evidence(ms):
        d.unhl()
        d.click(d.expander("Show evidence (4 runs)"), after=800)
        d.hl(d.vis(p.get_by_role("button", name="sky-0904-0028")))

    def act_open(ms):
        d.unhl()
        d.click(d.vis(p.get_by_role("button", name="sky-0904-0028")), after=1500)

    def det_score(ms):
        sb = d.vis(p.locator("[data-testid='stSelectbox']").filter(has_text="Choose a run to inspect"))
        d.scroll_to(sb, "start")
        d.focus(d.gauge(), scroll=False)
        d.wait(1200)
        d.tour([d.text("Tool errors: 3"), d.text("Actions claimed but not done: 1")], ms - 3000)

    def det_timeline(ms):
        d.focus(d.text("Timeline — what happened"), block="start")
        errs = d.page.get_by_text("update_contact FAILED").filter(visible=True)
        d.tour([errs.nth(0), errs.nth(1), errs.nth(2), p.locator(".arc-quote").filter(visible=True).nth(1)], ms - 1500)

    def det_findings(ms):
        d.focus(d.text("Findings (3)"), block="start")
        titles = [d.vis(p.locator("strong").filter(has_text=t)) for t in
                  ("Claimed an action that never happened", "Said “done” after a failed tool", "Retry loop")]
        d.tour(titles, ms - 1500)

    def hil_choose(ms):
        d.unhl()
        d.drift(int(ms * 0.35))
        d.choose_run("sky-0905-0301")

    def hil_refund(ms):
        box = d.logs_box()
        d.scroll_to(box)
        d.hl(box)
        d.tour([box.locator("li").filter(has_text="verify_customer").first,
                box.locator("li").filter(has_text="issue_refund").first], ms - 1200, hl=False)

    def hil_escalate(ms):
        d.focus(d.logs_box().locator("li").filter(has_text="escalate_to_human").first, scroll=False)
        d.drift(int(ms * 0.4))
        d.focus(d.vis(d.claims_box().locator(".arc-quote")), scroll=False)

    def hil_right(ms):
        d.drift(int(ms * 0.3))
        d.focus(d.gauge())
        d.drift(int(ms * 0.35))
        d.focus(d.text("No findings in this run."))

    def ladder_open(ms):
        d.unhl()
        d.scroll_to(d.vis(p.get_by_role("tab", name="Autonomy ladder", exact=True)), "center")
        d.tab("Autonomy ladder")
        rung = [d.vis(p.get_by_role("heading", name=n)) for n in ("Autonomous", "Human approves", "Suggest only")]
        d.scroll_to(rung[0], "start")
        d.tour(rung + [d.text("injected_instruction · 2 runs")], ms - 2600)

    def acc_open(ms):
        d.tab("Model accuracy")
        d.focus(d.text("How good was the success probability on this file?"))

    def acc_numbers(ms):
        d.tour([d.metric("AUC"), d.metric("claims that were false")], ms - 300)

    def acc_trained(ms):
        d.focus(d.text("About the success score"), block="start")
        d.focus(d.li("Trained on:"), scroll=False)

    def acc_transfer(ms):
        d.focus(d.li("Tested on an agent it never saw"), scroll=False)

    def build_shot(img: str, caption: str):
        def act(ms):
            d.unhl()
            show_overlay(d.page, ROOT / "docs/images" / img, caption, ms)
            d.move(W * 0.62, H * 0.55, steps=30)
            d.wait(int(ms * 0.4))
            d.move(W * 0.45, H * 0.5, steps=40)
        return act

    def build_tests(ms):
        build_shot("process_4_tests_and_run.png", "Tests pass · app running")(ms)

    def close_download(ms):
        hide_overlay(d.page)
        d.unhl()
        d.focus(d.vis(p.get_by_role("button", name="Download full report (Markdown)")))

    def close_verify(ms):
        title()
        d.drift(int(ms * 0.3))
        d.tour([d.text("Step 1 · Upload your agent logs"), d.text("Step 2 · Policy (optional)")], int(ms * 0.6))
        d.unhl()

    def close_credits(ms):
        title()
        show_card(d.page, "Agent Report Card", "All demo data is synthetic, generated by us · Built with Claude Code")

    acts = {k: v for k, v in locals().items() if callable(v) and k not in ("yaml_key", "title", "build_shot")}
    acts.update(build_idea=build_shot("process_1_idea.png", "Built with Claude Code · the idea"),
                build_prompt=build_shot("process_2_build_prompt.png", "One short build prompt"),
                build_streamlit=build_shot("process_3_redirect_to_streamlit.png", "Steering it: a Streamlit app with tests"))
    acts.update(policy_require=yaml_key("require_before", 4), policy_injection=yaml_key("injection_patterns", 6),
                policy_escalate=yaml_key("escalate_if_text_matches", 4), policy_caps=yaml_key("caps", 2))
    return acts


# ----------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="am_michael")
    ap.add_argument("--speed", type=float, default=1.1)
    ap.add_argument("--dry", action="store_true", help="no TTS, short holds: just test every beat's action")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads((HERE / "narration.json").read_text(encoding="utf-8"))
    beats = [b for s in scenes for b in s["beats"]]
    if args.dry:
        dur = {b["key"]: len(b["say"].split()) / 2.7 for b in beats}   # estimate, used to spread tours
        audio = {}
    else:
        audio = synth(beats, args.voice, args.speed)
        dur = {k: len(v) / SR for k, v in audio.items()}
        long = [k for k, v in dur.items() if v > MAX_BEAT]
        if long:
            print(f"!! beats longer than {MAX_BEAT}s: {long}")
    print("narration total: %.1fs over %d beats" % (sum(dur.values()) + GAP * len(beats), len(beats)))

    timeline = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=str(OUT / "raw_v2"),
                                  record_video_size={"width": W, "height": H}, color_scheme="light")
        ctx.add_init_script(OVERLAY_JS)
        page = ctx.new_page()
        d = Demo(page, args.dry)
        acts = build_actions(d)
        missing = [b["key"] for b in beats if b["key"] not in acts]
        assert not missing, f"no action for beats {missing}"
        t0 = time.monotonic()
        d.preroll()
        lead = time.monotonic() - t0
        for b in beats:
            key, ms = b["key"], int(dur[b["key"]] * 1000)
            start = time.monotonic()
            try:
                acts[key](ms)
            except Exception as e:  # keep recording; report which beat to fix
                d.failures.append(key)
                print(f"!! beat {key} failed: {type(e).__name__}: {str(e).splitlines()[0][:220]}")
            act_s = time.monotonic() - start
            end = start + (0.25 if args.dry else dur[key] + GAP)
            rem = int((end - time.monotonic()) * 1000)
            if rem > 2000:
                d.drift(rem)
            elif rem > 0:
                d.wait(rem)
            timeline.append(dict(key=key, start=round(start - t0 - lead, 2), action_s=round(act_s, 2),
                                 speech_s=round(dur[key], 2), say=b["say"]))
            flag = "  <-- action longer than speech" if act_s > dur[key] else ""
            print(f"{start - t0 - lead:6.1f}s {key:<18} action {act_s:4.1f}s speech {dur[key]:4.1f}s{flag}")
        d.wait(1500)
        vpath = page.video.path()
        ctx.close()
        browser.close()

    print(f"failed beats: {len(d.failures)} {d.failures}")
    if args.dry:
        Path(vpath).unlink(missing_ok=True)
        return 1 if d.failures else 0

    total = time.monotonic() - t0 - lead
    track = np.zeros(int((total + 1) * SR), dtype=np.float32)
    for t in timeline:
        a = audio[t["key"]]
        i = int(t["start"] * SR)
        track[i:i + len(a)] += a[: max(0, len(track) - i)]
    wav = OUT / "narration_track.wav"
    sf.write(wav, track, SR)
    final, silent = OUT / "agent_report_card_demo_v2.mp4", OUT / "agent_report_card_demo_v2_silent.mp4"
    vc = ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "25"]
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{lead:.2f}", "-i", str(vpath), "-i", str(wav),
                    "-map", "0:v", "-map", "1:a", *vc, "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-t", f"{total:.2f}",
                    str(final)], check=True)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(final), "-c:v", "copy", "-an", str(silent)],
                   check=True)

    def ts(x: float) -> str:
        return "%02d:%02d:%02d,%03d" % (x // 3600, x % 3600 // 60, x % 60, (x * 1000) % 1000)
    (OUT / "agent_report_card_demo_v2.srt").write_text("\n".join(
        f"{i}\n{ts(t['start'])} --> {ts(t['start'] + t['speech_s'])}\n{t['say']}\n"
        for i, t in enumerate(timeline, 1)), encoding="utf-8")
    (OUT / "beat_timeline.json").write_text(json.dumps(dict(total_s=round(total, 1), beats=timeline), indent=1))

    FRAMES.mkdir(parents=True, exist_ok=True)
    for f in FRAMES.glob("*.png"):
        f.unlink()
    for i, t in enumerate(timeline):
        at = t["start"] + min(t["action_s"] + 0.3, t["speech_s"])   # the screen as it looks once the beat's action is done
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{at:.2f}", "-i", str(final), "-frames:v", "1",
                        "-vf", "scale=960:-1", str(FRAMES / f"{i:02d}_{t['key']}.png")], check=True)
    print(f"done: {final} ({total:.1f}s), frames in {FRAMES}")
    return 1 if d.failures else 0


if __name__ == "__main__":
    sys.exit(main())
