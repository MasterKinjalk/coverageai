"""Render the README animations (GIF) and stills from the app's own views.

    uv run --no-sync python experiments/readme_media.py                 # everything
    uv run --no-sync python experiments/readme_media.py --only field lab

Outputs go to Docs/media/:

* field_run.gif      the "Field run" canvas replaying a closed-loop pass (gusty_afternoon, seed 1,
                     medium mismatch, model + card feedback), ending on the pass summary
* spray_lab.gif      the spray-lab rear view: droplets leave the boom, fall, evaporate and drift
* nozzle_compare.gif same weather, two nozzles: fine XR droplets vs air-induction ULD droplets
* wsp_cards.gif      the water-sensitive-paper card the camera sees for each section, with the
                     B0 reader's mask and predicted / measured / true coverage
* app_field_run.png  screenshot of the Streamlit app (Field run tab)

The canvas pages are the same HTML the Streamlit app embeds, filled with the same payloads and
opened in headless Chrome/Edge through the DevTools protocol. Each frame sets the animation time
directly (no wall clock), so the GIFs are reproducible. Frames are joined with ffmpeg
(palettegen/paletteuse). Needs a Chromium browser, ffmpeg on PATH and the `websockets` package.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Docs" / "media"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_screenshots import find_browser, free_port  # noqa: E402

SCENARIO = dict(preset="gusty_afternoon", seed=1, mismatch="med", policy="fast")


# ------------------------------------------------------------------ browser

class Browser:
    """Headless Chromium driven over CDP (one page)."""

    def __init__(self, width: int = 1200, height: int = 800):
        exe = find_browser()
        if exe is None:
            raise SystemExit("No Chrome/Edge found (see BROWSERS in demo_screenshots.py).")
        self.port = free_port()
        self.prof = tempfile.mkdtemp(prefix="cdp_")
        self.proc = subprocess.Popen(
            [exe, "--headless=new", "--hide-scrollbars", "--force-device-scale-factor=1",
             f"--remote-debugging-port={self.port}", f"--user-data-dir={self.prof}",
             "--allow-file-access-from-files", f"--window-size={width},{height}", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.ws_url = None
        for _ in range(40):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json", timeout=1))
                self.ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t.get("type") == "page")
                break
            except Exception:
                time.sleep(0.5)
        if self.ws_url is None:
            self.close()
            raise SystemExit("Browser did not expose a DevTools page.")
        self.ws = None
        self.n = 0

    async def __aenter__(self):
        import websockets
        self.ws = await websockets.connect(self.ws_url, max_size=None)
        await self.call("Page.enable")
        return self

    async def __aexit__(self, *exc):
        await self.ws.close()
        self.close()

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        shutil.rmtree(self.prof, ignore_errors=True)

    async def call(self, method: str, **params):
        self.n += 1
        mid = self.n
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def js(self, expr: str):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"JS error in {expr[:80]!r}: {r['exceptionDetails']}")
        return r.get("result", {}).get("value")

    async def viewport(self, w: int, h: int) -> None:
        await self.call("Emulation.setDeviceMetricsOverride", width=int(w), height=int(h),
                        deviceScaleFactor=1, mobile=False)

    async def goto(self, url: str, ready: str = "true", timeout: float = 60.0) -> None:
        await self.call("Page.navigate", url=url)
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if await self.js(f"document.readyState === 'complete' && !!({ready})"):
                    return
            except RuntimeError:
                pass
            await asyncio.sleep(0.25)
        raise TimeoutError(url)

    async def shot(self, path: Path, clip: dict | None = None) -> None:
        params = {"format": "png"}
        if clip:
            params["clip"] = {**clip, "scale": 1}
        r = await self.call("Page.captureScreenshot", **params)
        path.write_bytes(base64.b64decode(r["data"]))


async def frame2(b: Browser) -> None:
    """Let the page's own requestAnimationFrame loop paint twice."""
    await b.js("new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")


# ------------------------------------------------------------------ ffmpeg

def to_gif(frames: Path, out: Path, fps: float, width: int, hold_last: float = 0.0,
           colors: int = 128, dither: str = "none") -> None:
    """frames/%04d.png -> optimized GIF (one shared palette; no dithering keeps the dark UI small)."""
    ff = shutil.which("ffmpeg")
    if ff is None:
        raise SystemExit("ffmpeg not found on PATH.")
    pad = f",tpad=stop_mode=clone:stop_duration={hold_last}" if hold_last > 0 else ""
    vf = (f"fps={fps}{pad},scale={width}:-1:flags=lanczos,split[a][b];"
          f"[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
          f"[b][p]paletteuse=dither={dither}:diff_mode=rectangle")
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "%04d.png"),
                    "-vf", vf, "-loop", "0", str(out)], check=True)
    print(f"wrote {out.relative_to(ROOT)}  ({out.stat().st_size / 1e6:.1f} MB)")


def fresh_dir(tmp: Path, name: str) -> Path:
    d = tmp / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


# ------------------------------------------------------------------ data

def run_pass():
    from coverageai import app_logic as al
    from coverageai.config import spec
    c_req = float(spec()["c_req_pct"])
    mods = al.build_modules_adapter("spray", "b0")
    conds = al.make_conditions(SCENARIO["preset"], SCENARIO["seed"])
    res = al.run_pass_wrapper(mods, conds, SCENARIO["policy"], SCENARIO["seed"], SCENARIO["mismatch"], c_req)
    return res, c_req


def field_html(res, c_req) -> str:
    from field_run_view import _HTML
    from coverageai.field_anim import field_animation_payload
    data = json.dumps(field_animation_payload(res, c_req)).replace("</", "<\\/")
    return _HTML.replace("__PAYLOAD__", data)


def lab_html(setting, conditions, target: float, title: str | None = None, compact: bool = False) -> str:
    from coverageai import spray_lab as sl
    res = sl.run_lab(setting, conditions, seed=7, target_pct=target)
    payload = json.dumps(res.particles, separators=(",", ":")).replace("</", "<\\/")
    html = (ROOT / "app" / "static_spray_lab.html").read_text(encoding="utf8")
    html = html.replace("/*__PAYLOAD__*/null", payload, 1)
    extra = ""
    if title:
        extra += f"document.querySelector('.title').textContent = {json.dumps(title)};"
    if compact:
        extra += ("document.querySelector('.ctrl').style.display='none';"
                  "document.querySelector('.legend').style.display='none';")
    if extra:
        html = html.replace("</body>", f"<script>{extra}</script></body>")
    return html


# ------------------------------------------------------------------ recorders

async def rec_field(b: Browser, tmp: Path, res, c_req) -> None:
    page = tmp / "field_run.html"
    page.write_text(field_html(res, c_req), encoding="utf8")
    await b.viewport(1200, 760)
    await b.goto(page.as_uri(), "typeof draw === 'function'")
    frames = fresh_dir(tmp, "field_frames")
    n = int(await b.js("N"))
    fps, seconds = 15, 14.0
    total = int(fps * seconds)
    await b.js("playing=false; sync();")
    for i in range(total + 1):
        t = n * i / total
        await b.js(f"t={t!r}; playing=false; sync(); draw(); readout(); summary();"
                   "if (t < N) $('play').textContent = 'Pause';")   # show the playing state
        await frame2(b)
        await b.shot(frames / f"{i:04d}.png")
    to_gif(frames, OUT / "field_run.gif", fps, 960, hold_last=4.0)


async def rec_lab(b: Browser, tmp: Path, html: str, name: str, width_px: int = 1000,
                  gif_width: int = 860, seconds: float = 7.0, fps: int = 15, gif: bool = True) -> None:
    page = tmp / f"{name}.html"
    page.write_text(html, encoding="utf8")
    await b.viewport(width_px, 900)
    await b.goto(page.as_uri(), "window.__sprayLab")
    h = int(await b.js("Math.ceil(document.querySelector('.lab').getBoundingClientRect().bottom) + 8"))
    await b.viewport(width_px, h)
    await frame2(b)
    t_final = float(await b.js("__sprayLab.payload.tFinal"))
    frames = fresh_dir(tmp, f"{name}_frames")
    total = int(fps * seconds)
    for i in range(total + 1):
        u = i / total
        t = t_final * u ** 1.6          # slow start: the release is the interesting part
        await b.js(f"__sprayLab.seek({t!r}); document.getElementById('play').textContent = 'Pause';"
                   "document.getElementById('slow').textContent = '';")   # GIF timing is not real time
        await frame2(b)
        await b.shot(frames / f"{i:04d}.png")
    if gif:
        to_gif(frames, OUT / f"{name}.gif", fps, gif_width, hold_last=3.0)


async def rec_compare(b: Browser, tmp: Path, target: float) -> None:
    """Two lab runs in the same hot, windy weather, stacked vertically."""
    from coverageai import spray_lab as sl
    from coverageai.contracts import Setting
    x = sl.make_conditions(5.0, 90.0, 32.0, 30.0, 12.0, 0.6)
    runs = [("compare_xr", Setting("XR", "03", 60.0),
             "XR 11003 at 60 psi: fine droplets, 5 m/s crosswind, 32 °C, 30 % RH"),
            ("compare_uld", Setting("ULD", "03", 60.0),
             "ULD 120-03 at 60 psi (air induction): coarse droplets, same weather")]
    for name, u, title in runs:
        await rec_lab(b, tmp, lab_html(u, x, target, title=title, compact=True), name,
                      width_px=1000, seconds=6.0, gif=False)
    # stack the two frame sets
    a, c = tmp / "compare_xr_frames", tmp / "compare_uld_frames"
    frames = fresh_dir(tmp, "compare_frames")
    from PIL import Image
    for p in sorted(a.glob("*.png")):
        top, bot = Image.open(p).convert("RGB"), Image.open(c / p.name).convert("RGB")
        w = max(top.width, bot.width)
        im = Image.new("RGB", (w, top.height + bot.height + 6), (14, 17, 23))
        im.paste(top, (0, 0))
        im.paste(bot, (0, top.height + 6))
        im.save(frames / p.name)
    to_gif(frames, OUT / "nozzle_compare.gif", 15, 820, hold_last=3.5)


def rec_cards(tmp: Path, res, c_req: float, every: int = 1) -> None:
    """WSP card per section with the B0 reader's mask and the coverage numbers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from coverageai import app_logic as al

    bg, panel, txt, mut = "#0e1117", "#161b22", "#e6edf3", "#8a94a6"
    cols = {"predicted": "#6ea8fe", "after feedback": "#c58bff", "card reads": "#ffb84d", "true": "#e6edf3"}
    frames = fresh_dir(tmp, "card_frames")
    ks = list(range(0, res.n, every))
    for i, k in enumerate(ks):
        cv = al.replay_card(res, k)
        lg = res.logs[k]
        if cv.pred_mask is None:
            raise SystemExit(f"section {k + 1}: the B0 reader returned no mask")
        mask = cv.pred_mask
        raw = al.overlay_mask(cv.image, mask, alpha=0.0, max_width=900)
        ovl = al.overlay_mask(cv.image, mask, color=(0, 255, 200), alpha=0.6, max_width=900)
        fig = plt.figure(figsize=(9.6, 4.6), dpi=100, facecolor=bg)
        gs = fig.add_gridspec(2, 2, width_ratios=[2.8, 1.3], height_ratios=[1, 1],
                              left=0.02, right=0.98, top=0.86, bottom=0.13, wspace=0.10, hspace=0.18)
        for r, (img, lab) in enumerate(((raw, "card as imaged"), (ovl, "stains found by the B0 reader"))):
            ax = fig.add_subplot(gs[r, 0])
            ax.imshow(img)
            ax.set_axis_off()
            ax.text(0.01, 0.97, lab, transform=ax.transAxes, va="top", fontsize=9.5, color="#0e1117",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#e6edf3", ec="none", alpha=0.85))
        ax = fig.add_subplot(gs[:, 1], facecolor=panel)
        vals = {"predicted": lg.pred.coverage_pct, "after feedback": lg.corrected_pred_cov,
                "card reads": lg.meas.coverage_pct, "true": lg.outcome.coverage_pct}
        names = list(vals)[::-1]
        ax.barh(names, [vals[nm] for nm in names], color=[cols[nm] for nm in names], height=0.6)
        ax.axvline(c_req, color="#3ddc97", ls="--", lw=1.4)
        ax.text(c_req, len(names) - 0.35, f" target {c_req:.0f}%", color="#3ddc97", fontsize=8.5)
        for j, nm in enumerate(names):
            ax.text(vals[nm] + 0.8, j, f"{vals[nm]:.1f}%", va="center", color=txt, fontsize=9)
        ax.set_xlim(0, max(60.0, max(vals.values()) * 1.25))
        ax.tick_params(colors=mut, labelsize=9)
        for s in ax.spines.values():
            s.set_color("#2b3340")
        ax.set_xlabel("card coverage (%)", color=mut, fontsize=9)
        ax.set_title("coverage", color=txt, fontsize=10, loc="left")
        u, x = lg.u, lg.x
        fig.text(0.02, 0.95, f"Section {k + 1} / {res.n}", color=txt, fontsize=14, weight="bold", va="center")
        fig.text(0.21, 0.95, f"wind {x.wind_speed_ms:.1f} m/s   {x.temp_c:.0f} °C   {x.rh_pct:.0f}% RH   "
                 f"nozzle {u.family} {u.size} @ {u.pressure_psi:g} psi   feedback {lg.bias:+.1f} pp",
                 color=mut, fontsize=10, va="center")
        fig.savefig(frames / f"{i:04d}.png", facecolor=bg)
        plt.close(fig)
        print(f"card {k + 1}/{res.n}", end="\r")
    print()
    to_gif(frames, OUT / "wsp_cards.gif", 2, 900, hold_last=1.0, colors=192, dither="bayer:bayer_scale=4")


def app_screenshot(tmp: Path) -> None:
    """Full Streamlit app, Field run tab, after the pass has loaded."""
    sp = free_port()
    st_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "main.py"), "--server.headless", "true",
         "--server.port", str(sp), "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(90):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{sp}/_stcore/health", timeout=1)
                break
            except Exception:
                time.sleep(1)
        q = "&".join(f"{k}={v}" for k, v in {**SCENARIO, "verifier": "b0"}.items())

        async def go():
            async with Browser(1500, 1100) as b:
                await b.viewport(1500, 1100)
                await b.goto(f"http://127.0.0.1:{sp}/?{q}", "document.querySelector('iframe')", 240)
                await asyncio.sleep(8.0)
                await b.shot(OUT / "app_field_run.png")
        asyncio.run(go())
        print("wrote", (OUT / "app_field_run.png").relative_to(ROOT))
    finally:
        st_proc.terminate()


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=["field", "lab", "compare", "cards", "app"],
                    choices=["field", "lab", "compare", "cards", "app"])
    ap.add_argument("--tmp", type=Path, default=None, help="frame directory (default: a temp dir)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = args.tmp or Path(tempfile.mkdtemp(prefix="readme_media_"))
    tmp.mkdir(parents=True, exist_ok=True)

    from coverageai import spray_lab as sl
    from coverageai.config import spec
    from coverageai.contracts import Setting
    target = float(spec()["c_req_pct"])
    res = c_req = None
    if {"field", "cards"} & set(args.only):
        print("running the pass ...")
        res, c_req = run_pass()

    async def canvases():
        async with Browser() as b:
            if "field" in args.only:
                await rec_field(b, tmp, res, c_req)
            if "lab" in args.only:
                x = sl.make_conditions(4.0, 90.0, 28.0, 40.0, 12.0, 0.55)
                html = lab_html(Setting("XR", "03", 40.0), x, target)
                await rec_lab(b, tmp, html, "spray_lab", seconds=8.0)
            if "compare" in args.only:
                await rec_compare(b, tmp, target)
    if {"field", "lab", "compare"} & set(args.only):
        asyncio.run(canvases())
    if "cards" in args.only:
        rec_cards(tmp, res, c_req)
    if "app" in args.only:
        app_screenshot(tmp)
    if args.tmp is None:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
