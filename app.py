"""Streamlit demo: threshold image sharing for secure exam-paper distribution.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import io
import math
import time
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from sss_exam import metrics as mt
from sss_exam import scheme as sc
from sss_exam.sample import BLANK_FRAC, DEFAULT_ROI_FRAC, Q5_FRAC, exam_paper, fake_notice

st.set_page_config(page_title="Exam Paper Vault · Threshold Image Sharing", page_icon="🔐", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.hero {padding: 1.1rem 1.4rem; border-radius: 14px; margin-bottom: .6rem;
       background: linear-gradient(120deg, rgba(30,64,175,.12), rgba(13,148,136,.10));
       border: 1px solid rgba(30,64,175,.18);}
.hero h1 {font-size: 1.75rem; margin: 0 0 .25rem 0;}
.hero p {margin: 0; opacity: .85;}
.banner {padding: .85rem 1.1rem; border-radius: 10px; margin: .4rem 0 .9rem 0; font-size: 1.02rem;
         border-left: 6px solid;}
.banner b {font-size: 1.08rem;}
.ok   {background: rgba(22,163,74,.10);  border-color: #16a34a;}
.warn {background: rgba(217,119,6,.12);  border-color: #d97706;}
.bad  {background: rgba(220,38,38,.10);  border-color: #dc2626;}
.info {background: rgba(37,99,235,.09);  border-color: #2563eb;}
.chip {display:inline-block; padding: .12rem .55rem; border-radius: 999px; margin: 0 .25rem .25rem 0;
       font-size: .82rem; background: rgba(100,116,139,.15);}
.small {font-size: .86rem; opacity: .8;}
</style>
""", unsafe_allow_html=True)

DEFAULT_CUSTODIANS = ["Exam Board", "District Officer", "Centre Superintendent", "Police Observer",
                      "Secure Server", "Deputy Board", "Flying Squad", "Backup Vault"]
PALETTE = [(220, 38, 38), (234, 88, 12), (147, 51, 234), (8, 145, 178),
           (219, 39, 119), (101, 163, 13), (37, 99, 235), (120, 113, 108)]
ss = st.session_state


# ----------------------------------------------------------------------------- helpers
def rgb(img: np.ndarray) -> np.ndarray:
    return np.repeat(img, 3, axis=2) if img.shape[2] == 1 else img


def frac_to_rect(frac, shape):
    H, W = shape[:2]
    x0, y0, x1, y1 = frac
    return (int(round(x0 * W)), int(round(y0 * H)), int(round(x1 * W)), int(round(y1 * H)))


def overlay_rects(img, rects):
    """rects: list of (rect, colour, label)."""
    base = Image.fromarray(rgb(img)).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for (x0, y0, x1, y1), col, label in rects:
        d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=col + (38,), outline=col + (255,), width=3)
        if label:
            d.rectangle([x0, max(0, y0 - 22), x0 + 9 * len(label) + 12, y0], fill=col + (235,))
            d.text((x0 + 6, max(0, y0 - 20)), label, fill="white")
    return np.asarray(Image.alpha_composite(base, layer).convert("RGB"))


def tamper_overlay(rec: sc.Reconstruction, names: dict[int, str]):
    out = rgb(rec.image).astype(np.float32) * 0.45 + 255 * 0.55
    for i, x in enumerate(rec.xs):
        m = rec.share_error_maps[x]
        if m.any():
            out[m] = np.array(PALETTE[(x - 1) % len(PALETTE)], dtype=np.float32)
    if rec.uncorrectable_map.any():
        hatch = rec.uncorrectable_map & ((np.add.outer(np.arange(out.shape[0]), np.arange(out.shape[1])) // 6) % 2 == 0)
        out[hatch] = (220, 38, 38)
    return out.astype(np.uint8)


def kb(nbytes: float) -> str:
    return f"{nbytes / 1024:,.1f} KB"


def banner(kind: str, html: str):
    st.markdown(f'<div class="banner {kind}">{html}</div>', unsafe_allow_html=True)


def load_upload(file, max_side=1000, gray=False) -> np.ndarray:
    im = Image.open(file)
    im = im.convert("L" if gray else "RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    arr = np.asarray(im)
    return arr[:, :, None] if arr.ndim == 2 else arr


def predicted_share_bytes(params: sc.Params) -> int:
    H, W, C = params.shape
    roi = int(sc.roi_pixel_mask(params).sum()) * C
    bg = H * W * C - roi
    return math.ceil(roi / params.L_roi) + math.ceil(bg / params.L_bg)


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Scheme settings")
    n = st.slider("Custodians (n)", 3, sc.MAX_SHARES, 5, help="Number of shares created.")
    k = st.slider("Threshold (k)", 2, n, min(3, n), help="Any k shares rebuild the paper; k−1 learn nothing.")
    st.markdown("**Data per polynomial (L)**  \n<span class='small'>L=1 is Shamir (perfect secrecy, big shares); "
                "L=k is Thien–Lin (1/k size, leaks). In between is a ramp scheme.</span>", unsafe_allow_html=True)
    L_roi = st.select_slider("L for question region (ROI)", options=list(range(1, k + 1)), value=1)
    L_bg = st.select_slider("L for header / background", options=list(range(1, k + 1)), value=max(1, k - 1))
    gray = st.toggle("Grayscale paper (3× less data)", value=False)
    with st.expander("Custodian names"):
        names = [st.text_input(f"Share x={i + 1}", DEFAULT_CUSTODIANS[i], key=f"cust_{i}") for i in range(n)]
    st.divider()
    st.caption("Field: GF(2⁸), polynomial x⁸+x⁴+x³+x²+1 (same as `galois.GF(2**8)`). "
               "Randomness: `secrets` (CSPRNG).")

# ----------------------------------------------------------------------------- header
st.markdown("""
<div class="hero">
<h1>🔐 Exam Paper Vault</h1>
<p>A size-efficient, cheater-identifying threshold image sharing scheme for secure examination paper distribution.
Split the paper among <i>n</i> custodians · any <i>k</i> rebuild it on exam morning · fewer learn nothing ·
forged shares are caught and named.</p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["📄 1 · Paper & split", "🧩 2 · Shares", "🏫 3 · Exam morning",
                "🔍 4 · Leakage study", "♻️ 5 · Refresh", "📊 6 · Metrics"])

# ----------------------------------------------------------------------------- tab 1
with tabs[0]:
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        src = st.radio("Question paper", ["Sample paper", "Upload an image"], horizontal=True)
        if src == "Upload an image":
            up = st.file_uploader("PNG or JPG (resized to ≤1000 px)", type=["png", "jpg", "jpeg"])
            img = load_upload(up, gray=gray) if up else None
            is_sample = False
        else:
            img = exam_paper()
            if gray:
                img = np.asarray(Image.fromarray(img).convert("L"))[:, :, None]
            is_sample = True
        if img is None:
            st.info("Upload a scanned question paper to continue.")
            st.stop()

        st.markdown("**Question region (ROI)**: protected with its own L, usually 1 for perfect secrecy.")
        use_roi = st.toggle("Use a question region", value=True)
        roi = None
        if use_roi:
            d = DEFAULT_ROI_FRAC if is_sample else (0.0, 0.25, 1.0, 1.0)
            c1, c2 = st.columns(2)
            xr = c1.slider("Horizontal extent (%)", 0, 100, (int(d[0] * 100), int(d[2] * 100)))
            yr = c2.slider("Vertical extent (%)", 0, 100, (int(d[1] * 100), int(round(d[3] * 100))))
            roi = frac_to_rect((xr[0] / 100, yr[0] / 100, xr[1] / 100, yr[1] / 100), img.shape)
            if roi[2] <= roi[0] or roi[3] <= roi[1]:
                roi = None

    params = sc.Params(n=n, k=k, L_roi=L_roi, L_bg=L_bg, shape=img.shape, roi=roi)
    img_key = (img.shape, hash(img.tobytes()))

    with right:
        boxes = [(roi, (220, 38, 38), f"ROI  L={L_roi}")] if roi else []
        st.image(overlay_rects(img, boxes), caption=f"{img.shape[1]}×{img.shape[0]} px · "
                 f"{'grayscale' if img.shape[2] == 1 else 'RGB'} · {kb(img.size)} raw", width="stretch")

    with left:
        orig = img.size
        per = predicted_share_bytes(params)
        st.markdown("**Storage before splitting**")
        comp = pd.DataFrame({
            "Scheme": ["Plain Shamir (L=1 everywhere)", f"Thien–Lin (L={k} everywhere)",
                       "This configuration", "AES + Shamir on the 32-byte key"],
            "Each share": [kb(orig), kb(math.ceil(orig / k)), kb(per), "32 B"],
            "All n shares": [f"{n:.2f}×", f"{n / k:.2f}×", f"{n * per / orig:.2f}×", f"≈1.00× (+ ciphertext)"],
            "Secrecy against k−1": ["perfect", "leaks layout", "perfect in ROI" if L_roi == 1 else "partial",
                                    "computational (AES)"],
        })
        st.dataframe(comp, hide_index=True, width="stretch")

        if st.button(f"🔐 Split into {n} shares", type="primary", width="stretch"):
            t = time.perf_counter()
            shares = sc.split(img, params, names)
            ss.split_time = time.perf_counter() - t
            ss.update(img=img, img_key=img_key, params=params, shares=shares, old_shares=None,
                      grids={s.x: sc.share_to_grid(s) for s in shares}, rec=None, refresh_info=None)
            st.success(f"Created {n} shares in {ss.split_time * 1000:.0f} ms. Open tab 2.")

ready = "shares" in ss and ss.get("shares")
stale = ready and (ss.params != params or ss.img_key != img_key)


def need_shares():
    if not ready:
        st.info("Split the paper in tab 1 first.")
        return True
    if stale:
        st.warning("Settings or paper changed since the last split. The tabs below use the old shares; "
                   "click **Split** in tab 1 to update.")
    return False


def custodian_name(x):
    return next((s.custodian for s in ss.shares if s.x == x), f"x={x}")


# ----------------------------------------------------------------------------- tab 2
with tabs[1]:
    if not need_shares():
        P = ss.params
        st.markdown(f"Each custodian gets one share. It looks like random noise; on its own it reveals "
                    f"nothing about the questions. Any **{P.k} of {P.n}** rebuild the paper.")
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w") as z:
            for s in ss.shares:
                z.writestr(f"share_x{s.x}_{s.custodian.replace(' ', '_')}.png", s.to_png_bytes())
        st.download_button("⬇️ Download all shares (.zip)", zbuf.getvalue(), "exam_shares.zip", "application/zip")
        cols = st.columns(min(P.n, 4))
        for i, s in enumerate(ss.shares):
            with cols[i % len(cols)]:
                with st.container(border=True):
                    st.markdown(f"**{s.custodian}**  \n<span class='chip'>x = {s.x}</span>"
                                f"<span class='chip'>{kb(s.nbytes)}</span><span class='chip'>epoch {s.epoch}</span>",
                                unsafe_allow_html=True)
                    st.image(rgb(ss.grids[s.x]), width="stretch")
                    st.download_button("Download", s.to_png_bytes(), f"share_x{s.x}.png", "image/png",
                                       key=f"dl_{s.x}_{s.epoch}", width="stretch")
        st.caption("Shown painted back onto the page grid: in regions with L>1 one share byte covers L pixels, "
                   "so those parts look blockier. The downloadable file is the compact byte stream "
                   "with metadata (x, n, k, L, ROI, paper id) in a PNG text chunk.")

# ----------------------------------------------------------------------------- tab 3
ATTACKS = ["Honest", "Alter part of share", "Forge entire share", "Targeted forgery (fake notice)",
           "Old share (stolen before refresh)"]

with tabs[2]:
    if not need_shares():
        P = ss.params
        st.markdown("Custodians meet at the exam centre. Choose who turns up and whether anyone cheats.")
        present = sorted(st.multiselect("Custodians present", [s.x for s in ss.shares], default=[s.x for s in ss.shares][:P.k],
                                 format_func=lambda x: f"{custodian_name(x)} (x={x})"))
        m = len(present)
        t = (m - P.k) // 2
        chips = (f"<span class='chip'>{m} shares present</span><span class='chip'>threshold k = {P.k}</span>"
                 + (f"<span class='chip'>detects up to {m - P.k} bad</span><span class='chip'>corrects up to {t} bad</span>"
                    if m > P.k else ""))
        st.markdown(chips, unsafe_allow_html=True)

        opts = ATTACKS if ss.get("old_shares") else ATTACKS[:-1]
        behaviour = {}
        if present:
            cols = st.columns(min(len(present), 4))
            for i, x in enumerate(present):
                behaviour[x] = cols[i % len(cols)].selectbox(custodian_name(x), opts, key=f"beh_{x}")

        is_sample = ss.img.shape[:2] == exam_paper().shape[:2]
        alter_rect = forge_rect = None
        if "Alter part of share" in behaviour.values():
            d = Q5_FRAC if is_sample else (0.1, 0.5, 0.9, 0.6)
            c1, c2 = st.columns(2)
            ax = c1.slider("Altered area: horizontal (%)", 0, 100, (int(d[0] * 100), int(d[2] * 100)), key="ax")
            ay = c2.slider("Altered area: vertical (%)", 0, 100, (int(d[1] * 100), int(round(d[3] * 100))), key="ay")
            alter_rect = frac_to_rect((ax[0] / 100, ay[0] / 100, ax[1] / 100, ay[1] / 100), ss.img.shape)
        if "Targeted forgery (fake notice)" in behaviour.values():
            st.markdown("<span class='small'>Targeted forgery (Tompa–Woll attack): the cheater shifts their share so "
                        "that, when exactly k shares are combined, a fake notice appears where they guess the "
                        "paper is blank. Possible only in regions with L = 1. The cheater can then undo the shift "
                        "and read the real paper while everyone else sees the fake one.</span>", unsafe_allow_html=True)
            d = BLANK_FRAC if is_sample else (0.1, 0.85, 0.9, 0.95)
            c1, c2 = st.columns(2)
            fx = c1.slider("Fake notice: horizontal (%)", 0, 100, (int(d[0] * 100), int(d[2] * 100)), key="fx")
            fy = c2.slider("Fake notice: vertical (%)", 0, 100, (int(d[1] * 100), int(d[3] * 100)), key="fy")
            forge_rect = frac_to_rect((fx[0] / 100, fy[0] / 100, fx[1] / 100, fy[1] / 100), ss.img.shape)

        if st.button("🔓 Reconstruct paper", type="primary", disabled=m == 0):
            by_x = {s.x: s for s in ss.shares}
            old_by_x = {s.x: s for s in (ss.get("old_shares") or [])}
            used, notes = [], []
            coalition = sorted(present)[:P.k]
            for x in present:
                b = behaviour[x]
                s = by_x[x]
                if b == "Alter part of share":
                    s = sc.tamper_random(s, alter_rect)
                elif b == "Forge entire share":
                    s = sc.forge_whole(s)
                elif b == "Targeted forgery (fake notice)":
                    s, targeted = sc.targeted_forgery(s, coalition, fake_notice(ss.img.shape, forge_rect))
                    if not targeted:
                        notes.append("Targeted forgery needs a region with L = 1; this configuration has none, "
                                     "so the cheat had no effect.")
                elif b.startswith("Old share"):
                    s = old_by_x[x]
                used.append(s)
            t0 = time.perf_counter()
            with st.spinner("Interpolating and running Reed–Solomon decoding…"):
                rec = sc.reconstruct(used)
            ss.rec = rec
            ss.rec_meta = dict(time=time.perf_counter() - t0, behaviour=dict(behaviour), notes=notes)

        rec = ss.get("rec")
        if rec is not None and set(rec.xs) == set(present):
            meta = ss.rec_meta
            correct = np.array_equal(rec.image, ss.img)
            cheaters = [x for x, b in meta["behaviour"].items() if b != "Honest"]
            nm = {x: custodian_name(x) for x in rec.xs}
            if rec.status == "insufficient":
                banner("bad", f"<b>Only {len(rec.xs)} of {rec.k} shares.</b> This is what a group of insiders "
                       f"below the threshold gets: noise. The paper stays secret.")
            elif rec.status == "unverified":
                if correct:
                    banner("warn", f"<b>Exactly k = {rec.k} shares: paper rebuilt, but it can't be verified.</b> "
                           "With no spare share there is nothing to cross-check. Bring one more custodian to verify.")
                else:
                    banner("bad", f"<b>Exactly k = {rec.k} shares and a cheater is present: the output is wrong "
                           "and nothing warns you.</b> This is the gap the Reed–Solomon check closes. "
                           "Add one more honest custodian and reconstruct again.")
            elif rec.status == "clean":
                banner("ok", f"<b>Verified ✓</b> All {len(rec.xs)} shares lie on the same polynomials. "
                       "No tampering anywhere on the page.")
            elif rec.status == "corrected":
                who = ", ".join(f"{nm[x]} (x={x})" for x in rec.guilty)
                banner("ok" if correct else "warn",
                       f"<b>Cheater identified: {who}.</b> The bad bytes were corrected using the honest shares, "
                       f"so the paper below is the genuine one{' (matches the original exactly)' if correct else ''}.")
            else:
                banner("bad", f"<b>Tampering detected, but too many bad shares to correct.</b> "
                       f"With {len(rec.xs)} shares the decoder corrects at most {rec.correct_limit}. "
                       "Hatched areas can't be trusted; bring more honest custodians.")
            for note in meta["notes"]:
                st.info(note)

            c1, c2 = st.columns(2, gap="large")
            c1.image(rgb(rec.image), caption="Reconstructed paper", width="stretch")
            if rec.status in ("corrected", "detected"):
                c2.image(tamper_overlay(rec, nm), caption="Tamper map: colour = guilty share, hatched = uncorrectable",
                         width="stretch")
            with c2:
                rows = []
                for x in rec.xs:
                    if rec.status in ("insufficient", "unverified"):
                        verdict = "– not checkable (need > k shares)"
                    elif rec.share_error_groups[x]:
                        area = rec.share_error_maps[x].mean() * 100
                        verdict = f"❌ cheated · {area:.1f}% of page"
                    elif rec.status == "detected":
                        verdict = "? unknown"
                    else:
                        verdict = "✅ consistent"
                    rows.append({"Custodian": nm[x], "x": x, "Actually did": meta["behaviour"].get(x, "Honest"),
                                 "System verdict": verdict})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption(f"Decoding time: {meta['time'] * 1000:.0f} ms · "
                           f"output {'identical to' if correct else 'DIFFERENT from'} the original "
                           f"(PSNR {'∞' if correct else f'{mt.psnr(rec.image, ss.img):.1f} dB'})")

        with st.expander("📂 Reconstruct from share files instead"):
            files = st.file_uploader("Upload share PNGs", type=["png"], accept_multiple_files=True)
            if files and st.button("Reconstruct uploaded shares"):
                try:
                    loaded = [sc.Share.from_png_bytes(f.getvalue()) for f in files]
                    for w in sc.check_compatible(loaded):
                        st.warning(w)
                    r = sc.reconstruct(loaded)
                    st.write(f"Status: **{r.status}** · shares x = {r.xs}"
                             + (f" · guilty: {r.guilty}" if r.guilty else ""))
                    st.image(rgb(r.image), width="stretch")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Could not reconstruct: {e}")

# ----------------------------------------------------------------------------- tab 4
with tabs[3]:
    if not need_shares():
        P = ss.params
        st.markdown(
            f"**Why not just use Thien–Lin everywhere?** Exam papers are mostly white. With L = k there is no "
            f"randomness, so every white group gives the same share byte and the text layout shows through. "
            f"A ramp scheme with L data bytes keeps any **k − L** colluders completely blind, but **k − L + 1** "
            f"of them can cancel the randomness and see a fixed combination of the pixels.")
        c = st.slider("Number of colluding custodians", 1, P.k - 1, 1) if P.k > 2 else 1
        key = ("leak", ss.img_key, P.k, P.n)
        if ss.get("leak_key") != key:
            ss.leak_shares = {}
            for L in range(1, P.k + 1):
                pp = sc.Params(n=P.n, k=P.k, L_roi=1, L_bg=L, shape=ss.img.shape, roi=None)
                ss.leak_shares[L] = sc.split(ss.img, pp)
            ss.leak_key = key
        cols = st.columns(P.k)
        for L, col in zip(range(1, P.k + 1), cols):
            shares_L = ss.leak_shares[L]
            view = sc.collusion_view(shares_L[:c], "bg")
            label = "Shamir" if L == 1 else ("Thien–Lin" if L == P.k else "ramp")
            with col:
                st.markdown(f"**L = {L}** · {label} · share = 1/{L} size")
                if view is None:
                    st.image(rgb(sc.share_to_grid(shares_L[0])), width="stretch")
                    banner("ok", f"{c} colluder{'s' if c > 1 else ''}: <b>nothing leaks</b>. Shares are uniformly "
                           f"random (needs ≥ {P.k - L + 1} to learn anything).")
                else:
                    pic = sc.paint(shares_L[0].params, {"bg": view})
                    st.image(rgb(pic), width="stretch")
                    banner("bad", f"{c} colluder{'s' if c > 1 else ''}: <b>layout leaks</b>. "
                           f"Entropy {mt.entropy(view):.2f} bits/byte (8 = random).")
        st.caption("This motivates the ROI-adaptive design: L = 1 where the questions are, a larger L only for "
                   "the public header and instructions.")

# ----------------------------------------------------------------------------- tab 5
with tabs[4]:
    if not need_shares():
        P = ss.params
        st.markdown(
            "A custodian is transferred, or a share might have been copied. **Proactive refresh** adds shares of "
            "a random polynomial with zero data coefficients to every share. The paper is unchanged, but old shares "
            "no longer combine with new ones, so a stolen share becomes worthless. Nobody reconstructs the paper "
            "during the refresh.")
        if P.L_bg == P.k or P.L_roi == P.k:
            st.warning("A region with L = k has no random coefficients, so it **cannot be refreshed**. "
                       "That's another cost of Thien–Lin.")
        if st.button(f"♻️ Refresh all shares (epoch {ss.shares[0].epoch} → {ss.shares[0].epoch + 1})", type="primary"):
            old = ss.shares
            new, info = sc.refresh(old)
            ss.update(old_shares=old, shares=new, grids={s.x: sc.share_to_grid(s) for s in new},
                      refresh_info=info, rec=None)
            st.success("Shares refreshed. In tab 3 you can now pick “Old share (stolen before refresh)”.")
        if ss.get("old_shares"):
            x = st.selectbox("Compare custodian", [s.x for s in ss.shares], format_func=custodian_name)
            o = next(s for s in ss.old_shares if s.x == x)
            nw = next(s for s in ss.shares if s.x == x)
            c1, c2, c3 = st.columns(3)
            c1.image(rgb(sc.share_to_grid(o)), caption=f"Before (epoch {o.epoch})", width="stretch")
            c2.image(rgb(ss.grids[x]), caption=f"After (epoch {nw.epoch})", width="stretch")
            others = [s for s in ss.shares if s.x != x][:P.k - 1]
            mixed = sc.reconstruct([o] + others)
            c3.image(rgb(mixed.image), caption=f"Old share of x={x} + {P.k - 1} new shares", width="stretch")
            st.caption(f"Refreshed regions: {', '.join(r for r, v in ss.refresh_info.items() if v) or 'none'}.")

# ----------------------------------------------------------------------------- tab 6
with tabs[5]:
    if not need_shares():
        P = ss.params
        rows = [{"Image": "Original paper", "Raw size": kb(ss.img.size), "PNG size": kb(mt.png_size(ss.img)),
                 "Entropy (bits/byte)": round(mt.entropy(ss.img), 3),
                 "Adjacent-pixel corr.": round(mt.adjacent_correlation(ss.img), 4),
                 "NPCR % vs original": None, "UACI % vs original": None}]
        for s in ss.shares:
            g = ss.grids[s.x]
            npcr, uaci = mt.npcr_uaci(ss.img, g)
            flat = np.concatenate([s.streams[r] for r in sc.REGIONS])
            rows.append({"Image": f"Share x={s.x} ({s.custodian})", "Raw size": kb(s.nbytes),
                         "PNG size": kb(len(s.to_png_bytes())), "Entropy (bits/byte)": round(mt.entropy(flat), 3),
                         "Adjacent-pixel corr.": round(mt.adjacent_correlation(g), 4),
                         "NPCR % vs original": round(npcr, 2), "UACI % vs original": round(uaci, 2)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        total = sum(s.nbytes for s in ss.shares)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total share storage", kb(total), f"{total / ss.img.size:.2f}× original", delta_color="off")
        c2.metric("vs plain Shamir", f"{(1 - total / (P.n * ss.img.size)) * 100:.0f}% smaller")
        c3.metric("Split time", f"{ss.split_time * 1000:.0f} ms")
        c4.metric("Recovery", "Lossless", "PSNR = ∞", delta_color="off")
        st.caption("Notes for the thesis: NPCR ≈ 99.6% is ideal for a uniformly random share. UACI ≈ 33.5% is the "
                   "ideal between two random images; against a mostly white page the expected value is ≈ 50%, "
                   "so report which reference you used. Shares don't compress (PNG ≈ raw), while the original "
                   "document compresses very well. That's why the AES + key-sharing baseline must be discussed honestly.")
