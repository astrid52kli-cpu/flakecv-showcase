"""flakecv viewer.   Start it with:   flakecv app
(or: streamlit run app/app.py)

Plain Streamlit.  Every figure comes from flakecv.figures, the same code
`flakecv figure ...` uses, so anything seen here can be regenerated headlessly.
"""
from __future__ import annotations

import hashlib
import io as _io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from flakecv import analysis as A, baseline, curve as C, figures as FG, io, labels as L, labelui as LU, model as Mdl
# The repository is found from this file, not from the installed package:
# on a non-editable install (e.g. Streamlit Cloud) the package lives in
# site-packages, away from data/ and models/.
REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO / "models" / "default.joblib"
from flakecv.config import DataRootError, allow_writes, data_root

st.set_page_config(page_title="flakecv", page_icon="🔬", layout="wide")


# ------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str, mtime: float):
    return Mdl.load(path)


@st.cache_data(show_spinner="Computing features and model probability…", max_entries=8)
def probability(key: str, _img, px_um, model_path, model_mtime):
    clf, meta = load_model(model_path, model_mtime)
    return Mdl.probability_image(clf, meta, _img, px_um)


@st.cache_data(show_spinner=False, max_entries=16)
def rules_mask(key: str, _img, px_um):
    return A.rules_mask(_img, px_um)


@st.cache_data(show_spinner=False, max_entries=8)
def image_url(key: str, _img):
    return LU.image_url(_img)


@st.cache_data(show_spinner="Preparing the starting label…", max_entries=8)
def prefill_label(key: str, _img, px_um):
    return L.prefill(_img, px_um)[0]


@st.cache_resource
def labeller():
    return LU.component()


def png_bytes(mask) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.png"
        L.save(p, np.asarray(mask, np.uint8))
        return p.read_bytes()


def fig_png(fig) -> bytes:
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor="white")
    return buf.getvalue()


def sop_warnings(px_um, hv_kV):
    w = []
    if hv_kV is not None and hv_kV < 20:
        w.append(f"Acquired at {hv_kV:g} kV. The SOP asks for 20 kV: at 5 kV the flake-on-gold "
                 "contrast collapses and can flip sign. Expect a worse mask.")
    if px_um > 0.070:
        w.append(f"Pixel size {px_um * 1000:.0f} nm is coarser than the SOP's 70 nm/px.")
    return w


# ------------------------------------------------------------------ sidebar
st.sidebar.title("flakecv")
try:
    root = data_root(start=REPO)
except DataRootError as e:
    root = None
    st.sidebar.warning(str(e))

model_path = st.sidebar.text_input("Model file", str(DEFAULT_MODEL))
try:
    clf, meta = load_model(model_path, Path(model_path).stat().st_mtime)
except FileNotFoundError:
    st.error(f"Model file not found: {model_path}. Train one with `flakecv train`.")
    st.stop()
except Mdl.ArtifactError as e:
    st.error(str(e))
    st.stop()
trained_on = {t["name"] for t in meta["training"]}

st.sidebar.subheader("1 · SEM frame")
src = st.sidebar.radio("Source", ["From the data folder", "Upload a file"], horizontal=True,
                       label_visibility="collapsed")
name = img = px = label = hv = sem_obj = up_raw = None
up_suffix = ".tiff"
if src == "From the data folder":
    frames = io.list_frames(root) if root else []
    if not frames:
        st.sidebar.info("No prepared frames in the data folder.")
    else:
        lab_names = [f for f in frames if io.frame_paths(root, f)["label"].is_file()]
        name = st.sidebar.selectbox("Frame", frames,
                                    format_func=lambda f: f + ("   ✓ label" if f in lab_names else ""))
        fr = io.load_frame(root, name)
        img, px, hv = fr.image, fr.px_um, fr.hv_kV
        lp = io.frame_paths(root, name)["label"]
        label = L.load(lp) if lp.is_file() else None
else:
    up = st.sidebar.file_uploader("SEM image", type=["tif", "tiff", "png", "jpg", "jpeg"])
    if up is not None:
        name = Path(up.name).stem
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / up.name
            p.write_bytes(up.getvalue())
            if p.suffix.lower() in (".tif", ".tiff"):
                try:
                    s = io.read_sem_tiff(p)
                    img, px, hv = s.image, s.px_um, s.hv_kV
                    sem_obj = s
                    up_raw, up_suffix = up.getvalue(), p.suffix.lower()
                    st.sidebar.caption(f"From the TIFF header: {px * 1000:.1f} nm/px, {hv} kV")
                except io.MetadataError as e:
                    st.sidebar.error(str(e))
            else:
                img = io.read_gray01(p)
                nm = st.sidebar.number_input("Pixel size (nm/px) — required for PNG/JPG",
                                             min_value=1.0, max_value=1000.0, value=None, step=1.0)
                px = None if nm is None else nm / 1000.0
                if px is not None:
                    sem_obj = io.SemFrame(image=io.quantize01(img), raw=img, px_um=px)
        upl = st.sidebar.file_uploader("Hand label (optional, 0/128/255 PNG)", type=["png"])
        if upl is not None and img is not None:
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "l.png"; p.write_bytes(upl.getvalue())
                label = L.load(p)
            if label.shape != img.shape:
                st.sidebar.error(f"Label is {label.shape}, image is {img.shape}; ignoring the label.")
                label = None

pp0 = meta["postproc"]
st.sidebar.subheader("2 · Post-processing")
thr = st.sidebar.slider("Probability threshold", 0.05, 0.95, float(pp0["thr"]), 0.05)
central = st.sidebar.checkbox("central: keep only the largest blob not touching the frame",
                              value=bool(pp0["central"]))
fill = st.sidebar.checkbox("fill: fill interior holes", value=bool(pp0["fill"]))
snap = st.sidebar.slider("snap radius (µm, 0 = off)", 0.0, 3.0, float(pp0["search_um"]), 0.25)
n_train = len(meta["training"])
if (central or fill) and n_train < 3:
    st.sidebar.warning("This model has fewer than 3 training frames: central/fill usually HURT here.")
if snap > 2.0:
    st.sidebar.caption("snap has its optimum at 1.5–2.0 µm and degrades by 3.0 µm.")

with st.sidebar.expander("About this model"):
    st.write(f"Feature set **{meta['feature_set']}** ({len(meta['feature_names'])} features), "
             f"trained on {n_train} frames, flakecv {meta['flakecv_version']}.")
    st.dataframe(pd.DataFrame(meta["training"])[["name", "px_um", "hv_kV"]], hide_index=True)
    st.json(meta["libraries"], expanded=False)

def dataset_panel(fname, from_folder):
    """Status of this frame in the dataset, plus adding/replacing its optical
    image and replacing its SEM image.  Messages survive the rerun."""
    if not fname:
        return
    fp = io.frame_paths(root, fname)
    have_sem, have_lab = fp["meta"].is_file(), fp["label"].is_file()
    om = LU.optical_for(root, fname)
    raws = LU.raw_paths(root, fname)
    tick = lambda b: "✅" if b else "⬜"
    with st.expander(f"Dataset entry **{fname}**:  {tick(have_sem)} SEM   {tick(have_lab)} label   "
                     f"{tick(om)} optical", expanded=not (have_sem and have_lab and om)):
        st.markdown(
            f"- SEM frame: {'`frames/' + fname + '.png`' if have_sem else 'not in the data folder yet (saved with the label, or below)'}"
            + (f"; raw TIFF `raw/{raws[0].name}`" if raws else "") + "\n"
            f"- hand label: {'`labels/' + fname + '.label.png`' if have_lab else 'none yet'}\n"
            f"- optical: " + (f"`{LU.PAIRED}/{om['file']}` (from {om['source_file']}, {om['objective']}"
                              + (f", {om['om_px_um']:.4f} µm/px)" if om.get('om_px_um') else ", pixel size unknown)")
                              if om else "none yet"))
        if not from_folder and not have_sem and sem_obj is not None:
            if st.button("Add this SEM frame to the data folder (without a label yet)"):
                r = LU.save_frame(root, fname, sem_obj, up_raw, up_suffix)
                st.session_state["ds_msg"] = (fname, f"Added SEM frame **{fname}**"
                                              + (f" and its raw TIFF `{r['raw'].name}`." if r["raw"] else "."))
                st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**" + ("Replace" if om else "Add") + " the optical image**")
            omf = st.file_uploader("Optical image", type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"],
                                   key=f"om_up_{fname}", label_visibility="collapsed")
            obj = st.radio("Objective", ["20x", "50x", "40x", "other"], horizontal=True, key=f"om_obj_{fname}")
            opx = io.OPTICAL_PX_UM.get(obj)
            if opx is None:
                opx = st.number_input("Pixel size (µm/px), blank if unknown", min_value=0.01, max_value=5.0,
                                      value=None, step=0.005, format="%.4f", key=f"om_px_{fname}")
            if st.button(("Replace" if om else "Save") + " optical image", disabled=omf is None,
                         key=f"om_save_{fname}"):
                try:
                    r = LU.save_optical(root, fname, omf.getvalue(), omf.name, obj, opx)
                    st.session_state["ds_msg"] = (fname, f"Saved optical image `{LU.PAIRED}/{r['file']}`"
                                                  + (f"; the old one is kept in `{r['backup']}`." if r["backup"] else "."))
                except Exception as e:  # noqa: BLE001
                    st.session_state["ds_msg"] = (fname, f"ERROR: {e}")
                st.rerun()
        with c2:
            if from_folder and have_sem:
                st.markdown("**Replace the SEM image** (e.g. a better re-shoot of the same flake)")
                sf = st.file_uploader("New SEM TIFF", type=["tif", "tiff"], key=f"sem_up_{fname}",
                                      label_visibility="collapsed")
                st.caption("The old frame, raw TIFF and label are copied to `data/backup/` first. The label is "
                           "then removed, because it was drawn on the old image: label the new one afterwards.")
                ok = st.checkbox("I understand the current label will be removed", key=f"sem_ok_{fname}")
                if st.button("Replace SEM image", disabled=sf is None or not ok, key=f"sem_rep_{fname}"):
                    with tempfile.TemporaryDirectory() as d:
                        tp = Path(d) / sf.name; tp.write_bytes(sf.getvalue())
                        try:
                            s2 = io.read_sem_tiff(tp)
                            r = LU.save_frame(root, fname, s2, sf.getvalue(), tp.suffix, replace=True)
                            st.session_state["ds_msg"] = (fname,
                                f"Replaced the SEM image of **{fname}** ({s2.px_um * 1000:.1f} nm/px, {s2.hv_kV} kV). "
                                f"Old files are in `{r['backup']}`."
                                + (" The old label was removed: label the new image now." if r["removed_label"] else ""))
                        except Exception as e:  # noqa: BLE001
                            st.session_state["ds_msg"] = (fname, f"ERROR: {e}")
                    st.cache_data.clear()
                    st.rerun()
        km = st.session_state.get("ds_msg")
        if km and km[0] == fname:
            (st.error if km[1].startswith("ERROR") else st.success)(km[1])


# ------------------------------------------------------------------ main
WRITES = allow_writes(start=REPO)
_names = ["Segmentation", "Optical → SEM"] + (["Label (hand)"] if WRITES else []) + \
         ["Defects (experimental)", "Learning curve"]
_tabs = dict(zip(_names, st.tabs(_names)))
tab_seg, tab_om, tab_def, tab_curve = (_tabs[n] for n in ("Segmentation", "Optical → SEM",
                                                         "Defects (experimental)", "Learning curve"))
tab_lab = _tabs.get("Label (hand)")

res = None
if img is not None and px is not None:
    key = f"{name}|{px}|{hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest()}"
    prob = probability(key, img, px, model_path, Path(model_path).stat().st_mtime)
    rm = rules_mask(key, img, px)
    mm = Mdl.finish_image(meta, prob, img, px, thr=thr, central=central, fill=fill, search_um=snap)
    sc = {} if label is None else dict(model=A.scores(mm, label, px), rules=A.scores(rm, label, px))
    res = dict(prob=prob, model=mm, rules=rm, scores=sc)

with tab_seg:
    if res is None:
        st.info("Pick a frame from the data folder or upload one in the sidebar."
                + (" A PNG/JPG also needs its pixel size." if img is not None else ""))
    else:
        for w in sop_warnings(px, hv):
            st.warning(w)
        if name in trained_on:
            st.warning(f"**{name}** is in this model's training set, so its scores are optimistic. "
                       "Honest numbers are on the Learning curve tab (leave-one-flake-out).")
        fig = FG.panels(name, img, px, res["rules"], res["model"], label, res["scores"])
        st.pyplot(fig, width="stretch")
        fig_bytes = fig_png(fig); plt.close(fig)
        area = float(res["model"].sum()) * px * px
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model flake area", f"{area:,.0f} µm²")
        c2.metric("Pixel size", f"{px * 1000:.1f} nm")
        if res["scores"]:
            s = res["scores"]["model"]
            c3.metric("IoU vs hand label", f"{s['iou']:.3f}",
                      f"{s['iou'] - res['scores']['rules']['iou']:+.3f} vs rules")
            c4.metric("Boundary F (0.25 µm)", f"{s['boundary_f']:.3f}")
            tbl = pd.DataFrame(A.metrics_rows(name, px, res)).set_index("method")
            st.dataframe(tbl[["iou", "dice", "boundary_f", "area_error", "area_pred_um2", "area_ref_um2"]]
                         .style.format("{:.3f}"), width="stretch")
        csv = pd.DataFrame(A.metrics_rows(name, px, res)).to_csv(index=False).encode()
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("Model mask (PNG)", png_bytes(res["model"]), f"{name}.mask.png", "image/png")
        d2.download_button("Rules mask (PNG)", png_bytes(res["rules"]), f"{name}.rules.png", "image/png")
        d3.download_button("Metrics (CSV)", csv, f"{name}.metrics.csv", "text/csv")
        d4.download_button("This figure (PNG)", fig_bytes, f"{name}.panels.png", "image/png")
        st.caption("Masks use the label format: 0 background, 128 flake. "
                   f"CLI equivalent: `flakecv figure panels {name}`")

with tab_om:
    if res is None:
        st.info("Pick a SEM frame first.")
    else:
        st.write("Registers the optical image to this SEM frame using the **model's** SEM mask as the "
                 "target (never the hand label), then shows all three in the same field at the same scale.")
        paired = LU.optical_for(root, name) if root else None
        opts = A.default_optical(root) if root else []
        if paired:
            opts = [paired["path"]] + opts
            st.markdown(f"This frame has its own optical image: **{paired['source_file']}** "
                        f"({paired['objective']}). Change it on the *Label (hand)* tab.")
        chosen = st.multiselect("Optical images to try (the best match is kept)", opts,
                                default=[paired["path"]] if paired else opts,
                                format_func=lambda p: Path(p).name)
        om_up = st.file_uploader("…or upload optical images", type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"],
                                 accept_multiple_files=True)
        objs = ["20x", "50x", "other"]
        obj0 = (paired["objective"] if paired and paired["objective"] in objs else "other") if paired else "20x"
        mag = st.radio("Objective", objs, index=objs.index(obj0), horizontal=True)
        if mag == "other":
            om_px = st.number_input("Optical pixel size (µm/px)", min_value=0.01, max_value=5.0, step=0.005,
                                    format="%.4f", value=(paired or {}).get("om_px_um"))
        else:
            om_px = io.OPTICAL_PX_UM[mag]
        st.caption("Optical scales are measured against SEM (the burnt-in scale bars are not calibrated): "
                   "20x = 0.2475 µm/px, 50x = 0.090 µm/px. 40x has not been calibrated yet.")
        optical = A.load_optical(chosen)
        for f in om_up or []:
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / f.name; p.write_bytes(f.getvalue())
                optical.update(A.load_optical([p]))
        rkey = (key, thr, central, fill, snap, om_px, tuple(sorted(optical)))
        if st.button("Register optical → SEM", type="primary", disabled=not optical or not om_px):
            with st.spinner("Searching mirror × rotation × translation… (about 1–4 minutes per optical image)"):
                st.session_state["reg"] = (rkey, A.register(img, px, res["model"], optical,
                                                            om_px=om_px, label=label))
        got = st.session_state.get("reg")
        if got and got[0] == rkey:
            r = got[1]
            if r is None:
                st.error("No optical flake could be matched to this frame.")
            else:
                fig3 = FG.threeway(name, r["om_warped"], r["mask"], img, px, res["model"], label, r["om"])
                st.pyplot(fig3, width="stretch")
                fig3_bytes = fig_png(fig3); plt.close(fig3)
                st.markdown(f"Matched **{r['om']}** · final stage **{r['stage']}** · "
                            f"{'mirrored' if r['mirrored'] else 'not mirrored'}")
                if "iou_vs_label" in r:
                    st.metric("Optical outline IoU vs hand label", f"{r['iou_vs_label']:.3f}")
                e1, e2 = st.columns(2)
                e1.download_button("Optical outline in SEM frame (PNG)", png_bytes(r["mask"]),
                                   f"{name}.optical_mask.png", "image/png")
                e2.download_button("This figure (PNG)", fig3_bytes, f"{name}.threeway.png", "image/png")
                st.caption(f"CLI equivalent: `flakecv figure threeway {name}`")
        elif got:
            st.caption("Settings changed since the last registration; press the button to run it again.")

with (tab_lab or st.empty()):
    if not WRITES:
        pass
    elif img is None or px is None:
        st.info("Pick a SEM frame from the data folder, or upload a SEM TIFF, in the sidebar.")
    elif root is None:
        st.error("No data folder is configured, so there is nowhere to save the label.")
    else:
        st.markdown("Paint the flake on the **SEM** image. Only the SEM frame is labelled: the model is "
                    "trained and scored (leave-one-flake-out) on SEM pixels against this label, so the "
                    "optical image is not needed here. It can still give you a good starting outline.")
        from_folder = src == "From the data folder"
        if from_folder:
            save_name = name
        else:
            save_name = st.text_input("Save as frame name", name).strip().replace("/", "-").replace(" ", "_")
            if save_name and io.frame_paths(root, save_name)["meta"].is_file():
                st.warning(f"A frame called **{save_name}** is already in the data folder. Saving adds a label "
                           "to THAT frame; it must be the same image. Pick another name, or use "
                           "*Replace the SEM image* there, if it is not.")
        dataset_panel(save_name, from_folder)
        starts = (["existing hand label"] if label is not None else []) + \
                 ["model mask", "grey-level prefill (bright = flake; erase the contacts)"]
        reg = st.session_state.get("reg")
        if reg and reg[0][0] == key and reg[1] is not None:
            starts.append("optical outline (from the Optical → SEM tab)")
        starts.append("blank")
        start = st.radio("Start from", starts, horizontal=True)
        band = 0.0
        if start in ("model mask", "optical outline (from the Optical → SEM tab)"):
            band = st.slider("uncertain band on the starting outline (µm)", 0.0, 1.0, 0.0, 0.25,
                             help="Pixels you cannot decide are excluded from scoring; leave 0 if you will "
                                  "trace the edge yourself.")
        if start == "existing hand label":
            init = label
        elif start == "model mask":
            init = LU.from_mask(res["model"], band, px)
        elif start.startswith("grey-level"):
            init = prefill_label(key, img, px)
        elif start.startswith("optical"):
            init = LU.from_mask(reg[1]["mask"], band, px)
        else:
            init = None
        token = f"{start}|{band}"
        st.caption("Changing *Start from* replaces what you painted. Unsaved strokes survive other "
                   "changes in the app, but not a page reload.")
        out = labeller()(key="labeller", data=dict(key=key, init_token=token, px_um=px,
                                                   image=image_url(key, img),
                                                   init=None if init is None else LU.label_url(init)),
                         on_saved_change=lambda: None)
        sv = getattr(out, "saved", None)
        sig = None if not sv else hashlib.sha1(str(sv.get("png")).encode()).hexdigest()
        if sv and sv.get("key") == key and save_name and sig != st.session_state.get("lab_sig"):
            st.session_state["lab_sig"] = sig
            try:
                new = LU.decode_label(sv["png"], img.shape)
                info = LU.save_label(root, save_name, new, None if from_folder else sem_obj,
                                     raw_bytes=None if from_folder else up_raw, raw_suffix=up_suffix)
                sc = A.scores(res["model"], new, px)
                st.session_state["lab_msg"] = (key,
                    f"Saved **{info['label'].name}** in `{info['label'].parent}`: flake "
                    f"{info['flake']:.1%}, uncertain {info['uncertain']:.1%} of the frame."
                    + (" The SEM frame was added to the data folder too." if info["wrote_frame"] else "")
                    + (f" The previous label is kept in `{info['backup']}`." if info["backup"] else "")
                    + f" Current model vs this label: IoU {sc['iou']:.3f}.")
            except Exception as e:  # noqa: BLE001 - shown to the user
                st.session_state["lab_msg"] = (key, f"ERROR: could not save the label: {e}")
            else:
                st.rerun()   # reload the sidebar so the new label is used everywhere
        km = st.session_state.get("lab_msg")
        m = km[1] if km and km[0] == key else None
        if m:
            (st.error if m.startswith("ERROR") else st.success)(m)
        st.caption("After labelling new frames: `flakecv train` retrains the default model and "
                   "`flakecv curve` recomputes the leave-one-flake-out curve with every labelled frame.")

with tab_def:
    st.warning("Step 3 is EXPERIMENTAL. No crack or bubble has been hand-labelled yet, so these are "
               "unvalidated candidates for review, not measurements.")
    if res is None:
        st.info("Pick a SEM frame first.")
    else:
        from flakecv import defects as D
        src_opts = (["hand label", "model mask"] if label is not None else ["model mask"])
        src_m = st.radio("Search inside", src_opts, horizontal=True)
        fmask = (label == 1).astype(np.uint8) if src_m == "hand label" else res["model"]
        dkey = (key, src_m, thr, central, fill, snap)
        if st.button("Find crack / bubble candidates", type="primary"):
            with st.spinner("Searching inside the flake… (about 10–60 s)"):
                st.session_state["def"] = (dkey, D.detect(img, fmask, px))
        got = st.session_state.get("def")
        if got and got[0] == dkey:
            dres = got[1]
            figd = FG.defects(name, img, px, fmask, dres)
            st.pyplot(figd, width="stretch")
            figd_bytes = fig_png(figd); plt.close(figd)
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Crack / fold chords (a consistent grey-level step across the chord)")
                if dres["cracks"]:
                    st.dataframe(pd.DataFrame(dres["cracks"])[["len_um", "snr", "agree", "score"]], width="stretch")
            with c2:
                st.caption("Bubble / residue spots (ranked; number = rank on the figure)")
                if dres["bubbles"]:
                    st.dataframe(pd.DataFrame(dres["bubbles"])[["r_um", "polarity", "score"]], width="stretch")
            import json as _json
            cand = dict(frame=name, px_um=px,
                        cracks=[dict(p0=c["p0"], p1=c["p1"]) for c in dres["cracks"]],
                        bubbles=[dict(x=b["x"], y=b["y"], r_px=b["r_px"]) for b in dres["bubbles"]])
            e1, e2 = st.columns(2)
            e1.download_button("Candidates (.defects.json)", _json.dumps(cand, indent=1).encode(),
                               f"{name}.defects.json", "application/json")
            e2.download_button("This figure (PNG)", figd_bytes, f"{name}.defects.png", "image/png")
            st.caption("The JSON is also the annotation format: delete the false ones, add the missed "
                       "ones, save it as data/defects/<frame>.defects.json, and `flakecv defects` "
                       f"will score against it. CLI equivalent: `flakecv defects {name}`")

with tab_curve:
    rp = st.text_input("Results file", str(REPO / "results" / "curve.json"))
    if not Path(rp).is_file():
        st.info("No results file yet. Create it with `flakecv curve` (about 30 minutes).")
    else:
        rows = C.load(rp)
        figc = FG.curve(rows, rules_mean=baseline.CURRENT["rules_iou_mean"])
        st.pyplot(figc, width="stretch"); plt.close(figc)
        S = C.summarise(rows)
        tbl = {lab: {f"n={n}": v[0] for n, v in S[k].items()}
               for k, lab in ((("photometric", "base"), "10 photometric"),
                              (("framing", "base"), "+ framing prior"),
                              (("framing", "snap"), "+ framing prior + snap"),
                              (("framing", "full"), "full post-processing")) if k in S}
        st.dataframe(pd.DataFrame(tbl).T.style.format("{:.3f}"), width="stretch")
        runs = {n: v[1] for n, v in next(iter(S.values())).items()}
        st.caption("Mean IoU, leave-one-flake-out on the six labelled flakes; runs per point "
                   + " / ".join(str(runs[n]) for n in sorted(runs))
                   + ". n=5 has one training set per held-out flake, so it is the noisiest point. "
                   "CLI equivalent: `flakecv figure curve`")
