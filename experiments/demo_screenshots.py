"""Capture screenshots of the Streamlit demo into experiments/figures/demo_*.png.

    uv run python experiments/demo_screenshots.py            # real browser screenshots
    uv run python experiments/demo_screenshots.py --static   # matplotlib stand-ins, no browser

Browser mode starts `streamlit run app/main.py` on a free port, drives a headless Chrome/Edge
through the DevTools protocol (needs the `websockets` package, a Streamlit dependency) and
saves four full-page shots: demo_live.png (segment 17 of gusty_afternoon, fast policy),
demo_field.png (lower half of the live tab), demo_policies.png (policy comparison) and
demo_nozzle.png (nozzle explorer). If no browser is found it falls back to --static, which
rebuilds the same panels with matplotlib from a scripted pass (no Streamlit involved).

Manual procedure (if both fail): `uv run streamlit run app/main.py`, press Run pass, step to
segment 17, and use the browser's "capture full size screenshot" on each tab.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "experiments" / "figures"
sys.path.insert(0, str(ROOT / "src"))

BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "google-chrome", "chromium", "chromium-browser", "microsoft-edge"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def find_browser() -> str | None:
    for b in BROWSERS:
        if Path(b).exists() or shutil.which(b):
            return b
    return None


# ------------------------------------------------------------------ CDP mode

async def _cdp_session(ws_url: str, app_url: str, out: Path, steps: int) -> None:
    import websockets
    async with websockets.connect(ws_url, max_size=None) as ws:
        n = 0

        async def call(method: str, **params):
            nonlocal n
            n += 1
            await ws.send(json.dumps({"id": n, "method": method, "params": params}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == n:
                    return msg.get("result", {})

        async def js(expr: str):
            r = await call("Runtime.evaluate", expression=expr, returnByValue=True)
            return r.get("result", {}).get("value")

        async def wait_for(text: str, timeout: float = 180.0):
            t0 = time.time()
            while time.time() - t0 < timeout:
                if await js(f"document.body.innerText.toLowerCase().includes({json.dumps(text.lower())})"):
                    return
                await asyncio.sleep(1.0)
            body = await js("document.body.innerText.slice(0, 400)")
            raise TimeoutError(f"{text!r}; page text: {body!r}")

        async def settle(t: float = 2.5):
            await asyncio.sleep(t)
            while await js("!!document.querySelector('[data-testid=stStatusWidget]')"):
                await asyncio.sleep(0.5)
            await asyncio.sleep(1.0)

        async def click(selector: str, text: str):
            return await js(
                "(()=>{const e=[...document.querySelectorAll(%s)].find(x=>x.innerText.trim().includes(%s));"
                "if(e){e.scrollIntoView();e.click();return true}return false})()" % (json.dumps(selector), json.dumps(text)))

        async def shot(name: str):
            h = await js("Math.max(document.querySelector('section.main, [data-testid=stMain]')"
                         ".scrollHeight, document.body.scrollHeight)") or 2400
            await call("Emulation.setDeviceMetricsOverride", width=1500, height=int(min(h + 60, 6000)),
                       deviceScaleFactor=1, mobile=False)
            await asyncio.sleep(1.0)
            r = await call("Page.captureScreenshot", format="png")
            (out / name).write_bytes(base64.b64decode(r["data"]))
            print("wrote", out / name)

        await call("Page.enable")
        await call("Emulation.setDeviceMetricsOverride", width=1500, height=2400, deviceScaleFactor=1,
                   mobile=False)
        await call("Page.navigate", url=app_url)
        await wait_for("Meeting C_req")
        await settle()
        for _ in range(steps):
            await click("button", "Step")
            await asyncio.sleep(0.4)
        await settle(4.0)
        await shot("demo_live.png")
        await js("document.querySelector('[data-testid=stMain]').scrollTo(0, 99999)")
        await asyncio.sleep(1.0)
        print("tab:", await click("[role=tab]", "Policy comparison"))
        await asyncio.sleep(2.0)
        print("run:", await click("button", "Run comparison"))
        await wait_for("A single seed is an anecdote", 300)
        await settle(4.0)
        await shot("demo_policies.png")
        await click("[role=tab]", "Nozzle explorer")
        await settle(4.0)
        await shot("demo_nozzle.png")
        await click("[role=tab]", "Live pass")
        await settle(3.0)
        await js("document.querySelector('[data-testid=stMain]').scrollTo(0, 99999)")
        await settle(2.0)
        await shot("demo_field.png")


def capture_browser(steps: int = 16) -> bool:
    browser = find_browser()
    try:
        import websockets  # noqa: F401
    except ImportError:
        browser = None
    if browser is None:
        return False
    sp, cp = free_port(), free_port()
    st_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "main.py"),
         "--server.headless", "true", "--server.port", str(sp), "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    prof = tempfile.mkdtemp(prefix="cdp_")
    ch = None
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{sp}/_stcore/health", timeout=1)
                break
            except Exception:
                time.sleep(1)
        ch = subprocess.Popen([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                               f"--remote-debugging-port={cp}", f"--user-data-dir={prof}",
                               "--window-size=1500,2400", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ws_url = None
        for _ in range(30):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{cp}/json", timeout=1))
                ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t.get("type") == "page")
                break
            except Exception:
                time.sleep(1)
        if ws_url is None:
            return False
        FIG.mkdir(parents=True, exist_ok=True)
        asyncio.run(_cdp_session(ws_url, f"http://127.0.0.1:{sp}/?preset=gusty_afternoon&mismatch=high&policy=fast&verifier=b0", FIG, steps))
        return True
    finally:
        for p in (ch, st_proc):
            if p is not None:
                p.terminate()
        shutil.rmtree(prof, ignore_errors=True)


# ---------------------------------------------------------------- static mode

def capture_static(seed: int = 1, step: int = 16) -> None:
    """Matplotlib stand-ins of the main panels from a scripted pass (no browser)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from coverageai import app_logic as al
    from coverageai.config import spec

    plt.rcParams.update({"figure.facecolor": "#0e1117", "axes.facecolor": "#161b22",
                         "axes.edgecolor": "#2b3340", "text.color": "#e6edf3",
                         "axes.labelcolor": "#e6edf3", "xtick.color": "#c9d1d9",
                         "ytick.color": "#c9d1d9", "grid.color": "#2b3340", "font.size": 9})
    FIG.mkdir(parents=True, exist_ok=True)
    c_req = float(spec()["c_req_pct"])
    mods = al.build_modules_adapter("oracle_nominal", "perfect")
    conds = al.make_conditions("gusty_afternoon", seed)
    res = al.run_pass_wrapper(mods, conds, "fast", seed, "high", c_req)
    k = step
    cv = al.replay_card(res, k)
    cols = al.CLASS_COLORS

    # --- live panel
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.2, 0.9])
    kp = al.compute_kpis(res.logs, c_req, upto=k)
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    lg = res.logs[k]
    d = lg.derived
    ax.text(0, 1, f"Segment {k + 1}   ({res.policy}, high mismatch)", fontsize=13, va="top", weight="bold")
    ax.text(0, 0.78, f"met {kp['n_met']}/{kp['n']}   mean cov {kp['mean_cov']:.1f}%   "
            f"mean vol {kp['mean_gpa']:.1f} GPA   drift {100 * kp['mean_drift']:.1f}%   "
            f"re-treat {kp['n_retreat']}", va="top")
    ax.text(0, 0.55, f"wind {lg.x.wind_speed_ms:.1f} m/s toward {lg.x.wind_dir_deg:.0f} deg, heading "
            f"{lg.x.heading_deg:.0f} deg (rel {al.wind_arrow_deg(lg.x):+.0f})\n"
            f"T {lg.x.temp_c:.1f} C, RH {lg.x.rh_pct:.0f}%, {lg.x.travel_speed_kmh:.1f} km/h, "
            f"boom {lg.x.boom_height_m:.2f} m", va="top")
    ax.text(0, 0.28, f"chosen {lg.u.family}-{lg.u.size} @ {lg.u.pressure_psi:g} psi: "
            f"{d.app_rate_gpa:.1f} GPA, class {d.droplet_class}, VMD {d.vmd_um:.0f} um", va="top")
    top = al.top_candidates_frame(res.info_for(k))
    if len(top):
        ax.text(0, 0.1, top.to_string(index=False), va="top", family="monospace", fontsize=6.5)
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(al.overlay_mask(cv.image, cv.gt_mask, color=(0, 255, 200)))
    ax.axis("off")
    ax.set_title(f"WSP card: true {lg.outcome.coverage_pct:.1f}%, measured {lg.meas.coverage_pct:.1f}%, "
                 f"predicted {lg.pred.coverage_pct:.1f}%", fontsize=9)
    ts = al.timeseries_frame(res.logs, upto=k)
    ax = fig.add_subplot(gs[1, 0])
    cc = spec()["coverage_classes"]
    ax.axhspan(0, cc["marginal_min"], color=cols["critical"], alpha=0.12)
    ax.axhspan(cc["marginal_min"], c_req, color=cols["marginal"], alpha=0.12)
    ax.axhspan(c_req, 100, color=cols["acceptable"], alpha=0.12)
    for name, c in (("predicted", "#6ea8fe"), ("corrected", "#c58bff"), ("measured", "#ffb84d"),
                    ("true", "#e6edf3")):
        ax.plot(ts["k"], ts[name], "o-" if name != "true" else "--", ms=3, color=c, label=name)
    ax.axhline(c_req, color="#3ddc97", ls=":")
    ax.set_xlim(1, res.n)
    ax.set_ylim(0, max(60, ts[["predicted", "corrected", "measured", "true"]].to_numpy().max() * 1.1))
    ax.set_xlabel("segment")
    ax.set_ylabel("coverage (%)")
    ax.legend(ncol=4, fontsize=7, frameon=False)
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(ts["k"], ts["residual"], color="#ffb84d", alpha=0.4, label="residual")
    ax.step(ts["k"], ts["bias"], where="post", color="#4dd0c8", label="applied bias")
    ax.axhline(0, color="#2b3340")
    ax.set_xlim(1, res.n)
    ax.set_xlabel("segment")
    ax.set_ylabel("pp")
    ax.legend(fontsize=7, frameon=False)
    ax = fig.add_subplot(gs[2, :])
    fs = al.field_strip_data(res.conds, res.logs, k, basis="measured")
    for _, r in fs.iterrows():
        ax.add_patch(plt.Rectangle((r["col"] + 0.04, -r["row"] + 0.04), 0.92, 0.92,
                                   color=cols[r["status"]], lw=0))
        ax.text(r["col"] + 0.5, -r["row"] + 0.5, str(int(r["seg"])), ha="center", va="center",
                fontsize=7, color="#0e1117")
        if r["retreat"]:
            ax.text(r["col"] + 0.82, -r["row"] + 0.85, "x", color="white", fontsize=9, weight="bold")
        if r["current"]:
            ax.add_patch(plt.Rectangle((r["col"] + 0.02, -r["row"] + 0.02), 0.96, 0.96, fill=False,
                                       ec="white", lw=2))
    ax.set_xlim(0, fs["col"].max() + 1)
    ax.set_ylim(-fs["row"].max(), 1)
    ax.axis("off")
    ax.set_title("Field strip (laps alternate direction; x = re-treatment flag)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "demo_live.png", dpi=130)
    plt.close(fig)

    # --- policy comparison
    out = al.compare_policies(mods, conds, seed, "high", c_req, None, ("open", "fast", "oracle"))
    tb = al.comparison_table(out)
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
    for a, col in zip(axs[:3], ("requirement met (%)", "mean volume (GPA)", "mean drift (frac)")):
        a.bar(tb["policy"], tb[col], color=["#6ea8fe", "#c58bff", "#3ddc97"])
        a.set_title(col, fontsize=9)
    for p, r in out.items():
        axs[3].plot(al.timeseries_frame(r.logs)["true"], label=p)
    axs[3].axhline(c_req, color="#3ddc97", ls=":")
    axs[3].legend(fontsize=7, frameon=False)
    axs[3].set_title("true coverage per segment", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "demo_policies.png", dpi=130)
    plt.close(fig)

    # --- nozzle explorer
    from coverageai.contracts import Conditions, Setting
    x = Conditions(3.0, 90.0, 0.0, 25.0, 55.0, 16.0, 0.5)
    fam, size = "XR", "04"
    lo, hi = al.nozzle_pressure_bounds(fam, size)
    u = Setting(fam, size, float((lo + hi) / 2))
    cvp = al.coverage_vs_pressure(fam, size, x)
    dvw = al.drift_vs_wind(u, x)
    grid = al.coverage_grid(fam, size, x)
    pat = al.pattern_overlap(al.nozzle_point(u, x)["derived"].spray_angle_deg, x.boom_height_m)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    axs[0].plot(cvp["pressure_psi"], cvp["coverage_pct"], color="#6ea8fe")
    axs[0].axhline(c_req, color="#3ddc97", ls=":")
    axs[0].set(title=f"{fam}-{size}: coverage vs pressure", xlabel="psi", ylabel="%")
    axs[1].plot(dvw["wind_ms"], dvw["drift_frac"], color="#ffb84d")
    axs[1].set(title="drift vs crosswind", xlabel="m/s")
    pv = grid.pivot(index="wind_ms", columns="pressure_psi", values="coverage_pct")
    im = axs[2].imshow(pv.to_numpy(), origin="lower", aspect="auto", cmap="turbo",
                       extent=[pv.columns.min(), pv.columns.max(), pv.index.min(), pv.index.max()])
    axs[2].set(title="coverage map (%)", xlabel="psi", ylabel="crosswind m/s")
    fig.colorbar(im, ax=axs[2])
    axs[3].plot(pat["x_m"] * 100, pat["single"], label="single nozzle")
    axs[3].plot(pat["x_m"] * 100, pat["total"], label="boom (sum)")
    axs[3].set(title=f"pattern overlap, uniformity {pat['uniformity']:.3f}", xlabel="cm")
    axs[3].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "demo_nozzle.png", dpi=130)
    plt.close(fig)
    print("wrote static demo_live/policies/nozzle.png in", FIG)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", action="store_true", help="matplotlib figures, no browser")
    ap.add_argument("--no-fallback", action="store_true", help="fail instead of using --static")
    ap.add_argument("--steps", type=int, default=16, help="segments to step before the live shot")
    args = ap.parse_args()
    if not args.static:
        try:
            if capture_browser(args.steps):
                return
        except Exception as e:  # fall back rather than leave the report without figures
            print("browser capture failed:", type(e).__name__, e)
        if args.no_fallback:
            raise SystemExit(1)
        print("falling back to static figures")
    capture_static(step=args.steps)


if __name__ == "__main__":
    main()
