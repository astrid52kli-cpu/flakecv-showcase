"""Hand-labelling inside the app: the browser canvas (an ``st.components.v2``
component) and the Python side that turns what it sends back into a label
file in the data folder.

The canvas paints the same three classes as ``docs/labeller.html`` and sends
the finished label as a 0/128/255 PNG (the disk format of ``labels.save``).
Only the SEM frame is labelled: the optical image is never needed for the
leave-one-flake-out curve, because the model is trained and scored on SEM
pixels against SEM labels.
"""
from __future__ import annotations

import base64
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from . import FLAKE, io, labels as L


# ------------------------------------------------------------------ encoding
def png_data_url(a: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", np.asarray(a))
    if not ok:
        raise ValueError("could not encode PNG")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def image_url(img01) -> str:
    return png_data_url(io.to_uint8(img01))


def label_url(lab) -> str:
    a = np.zeros(np.shape(lab), np.uint8)
    for cls, v in L.DISK.items():
        a[np.asarray(lab) == cls] = v
    return png_data_url(a)


def decode_label(url: str, shape) -> np.ndarray:
    """The canvas's PNG data URL -> in-memory classes, checked against the
    image shape."""
    if not isinstance(url, str) or "," not in url:
        raise ValueError("no label data received")
    raw = base64.b64decode(url.split(",", 1)[1])
    a = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise ValueError("label data is not a PNG")
    lab = L.from_disk(a)
    if lab.shape != tuple(shape[:2]):
        raise ValueError(f"label is {lab.shape}, image is {tuple(shape[:2])}")
    return lab


# ------------------------------------------------------------------ dataset
# One frame of the dataset = SEM frame (frames/) + its raw TIFF (raw/, optional)
# + hand label (labels/) + its optical image (optical_paired/).  Paired optical
# images are kept out of optical/ on purpose: the regression demo registers
# against every image in optical/, so adding files there would move its numbers.
PAIRED = "optical_paired"
IMAGE_EXT = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _backup(data_root, paths, stamp=None) -> Path | None:
    """Copy the existing files among ``paths`` to data/backup/<stamp>/,
    keeping their path relative to the data folder.  Returns the folder."""
    root = Path(data_root)
    have = [Path(p) for p in paths if Path(p).is_file()]
    if not have:
        return None
    dest = root / "backup" / (stamp or time.strftime("%Y%m%d-%H%M%S"))
    for p in have:
        q = dest / p.relative_to(root)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
    return dest


def raw_paths(data_root, name) -> list[Path]:
    d = Path(data_root) / "raw"
    return sorted(p for p in d.glob(f"{name}.*") if p.suffix.lower() in (".tif", ".tiff")) if d.is_dir() else []


def save_frame(data_root, name, sem: io.SemFrame, raw_bytes: bytes | None = None,
               raw_suffix: str = ".tiff", replace: bool = False) -> dict:
    """Add a SEM frame (prepared png + json, and the raw TIFF if given).
    With ``replace=True`` an existing frame is backed up first, and so is its
    label, which is then REMOVED: a label drawn on another image is not a label
    of this one."""
    p = io.frame_paths(data_root, name)
    if p["meta"].is_file() and not replace:
        raise FileExistsError(f"frame '{name}' is already in the data folder")
    old_raw = raw_paths(data_root, name)
    backup = _backup(data_root, [p["image"], p["meta"], p["label"], *old_raw]) if replace else None
    io.save_prepared(data_root, name, sem, extra=dict(source="app upload"))
    if replace:
        for q in old_raw:
            q.unlink()
    removed_label = False
    if replace and p["label"].is_file():
        p["label"].unlink()
        removed_label = True
    raw = None
    if raw_bytes:
        raw = Path(data_root) / "raw" / f"{name}{raw_suffix.lower()}"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(raw_bytes)
    return dict(name=name, backup=backup, removed_label=removed_label, raw=raw)


def save_label(data_root, name, lab, sem: io.SemFrame | None = None, raw_bytes=None,
               raw_suffix=".tiff") -> dict:
    """Write ``labels/<name>.label.png``.  An existing label is backed up
    first, never lost.  If the frame is not yet in the data folder, ``sem``
    (and the raw TIFF, if given) is added too."""
    p = io.frame_paths(data_root, name)
    wrote_frame = False
    if not p["meta"].is_file():
        if sem is None:
            raise FileNotFoundError(f"frame '{name}' is not in the data folder and no image was given")
        save_frame(data_root, name, sem, raw_bytes, raw_suffix)
        wrote_frame = True
    backup = _backup(data_root, [p["label"]])
    p["label"].parent.mkdir(parents=True, exist_ok=True)
    L.save(p["label"], lab)
    return dict(label=p["label"], backup=backup, wrote_frame=wrote_frame, **L.stats(lab))


def optical_for(data_root, name) -> dict | None:
    """The optical image paired with a frame: {path, objective, om_px_um, ...}."""
    d = Path(data_root) / PAIRED
    mp = d / f"{name}.om.json"
    if not mp.is_file():
        return None
    meta = json.loads(mp.read_text(encoding="utf-8"))
    path = d / meta["file"]
    if not path.is_file():
        return None
    return dict(meta, path=path, meta_path=mp)


def save_optical(data_root, name, data: bytes, filename: str, objective: str,
                 om_px_um: float | None) -> dict:
    """Pair an optical image with a frame (replacing, with backup, any image
    paired before)."""
    suffix = Path(filename).suffix.lower()
    if suffix not in IMAGE_EXT:
        raise ValueError(f"not an image file: {filename}")
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError(f"could not read {filename} as an image")
    if om_px_um is None and objective in io.OPTICAL_PX_UM:
        om_px_um = io.OPTICAL_PX_UM[objective]
    d = Path(data_root) / PAIRED
    old = optical_for(data_root, name)
    backup = _backup(data_root, [old["path"], old["meta_path"]]) if old else None
    if old:
        old["path"].unlink()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.om{suffix}"
    path.write_bytes(data)
    meta = dict(frame=name, file=path.name, source_file=filename, objective=objective,
                om_px_um=om_px_um, shape=list(arr.shape[:2]), added=time.strftime("%Y-%m-%d %H:%M:%S"))
    (d / f"{name}.om.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return dict(meta, path=path, backup=backup)


def from_mask(mask, uncertain_band_um=0.0, px_um=None) -> np.ndarray:
    """A binary mask as a starting label, optionally with an uncertain band
    of the given width on its outline."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    lab = np.where(m > 0, FLAKE, 0).astype(np.uint8)
    if uncertain_band_um and px_um:
        k = max(3, int(2 * uncertain_band_um / px_um) | 1)
        ker = np.ones((k, k), np.uint8)
        band = cv2.dilate(m, ker) - cv2.erode(m, ker)
        lab[band > 0] = 255
    return lab


# ------------------------------------------------------------------ canvas
CSS = r"""
:host{--flake:#1B9E8A;--uncert:#E0A32B;--back:#2F6FD0;--ink:#13181C;--dim:#5B666E;--line:#D9DDD7;--panel:#fff}
.root{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink)}
.bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:6px}
.bar button{border:1.5px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;
  padding:6px 10px;font-size:13px;font-weight:600;cursor:pointer;white-space:nowrap}
.bar button[aria-pressed="true"]{border-color:currentColor;box-shadow:inset 0 0 0 1px currentColor}
.bar .fl[aria-pressed="true"]{color:var(--flake)} .bar .bg[aria-pressed="true"]{color:var(--back)}
.bar .un[aria-pressed="true"]{color:var(--uncert)}
.bar .tbrush[aria-pressed="true"],.bar .tpoly[aria-pressed="true"],.bar .tlasso[aria-pressed="true"]{color:#ff5030}
.bar .done{background:#ff5030;border-color:#ff5030;color:#fff}
.bar .save{background:var(--flake);border-color:var(--flake);color:#fff}
.bar .dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:5px}
.bar .sep{width:1px;height:22px;background:var(--line);margin:0 3px}
.bar .val{font-variant-numeric:tabular-nums;font-size:12px;color:var(--dim);min-width:72px}
.bar input[type=range]{width:110px}
.stage{position:relative;height:72vh;min-height:420px;border:1px solid var(--line);border-radius:8px;
  overflow:hidden;background:#222}
canvas{display:block;cursor:crosshair;touch-action:none;image-rendering:pixelated}
.msg{font-size:12.5px;color:var(--dim);margin-top:5px;min-height:18px}
.keys{font-size:11.5px;color:var(--dim);margin-top:2px}
"""

HTML = ""

JS = r"""
const BG=0, FL=1, UN=2, DISK=[0,128,255];
const COL=[[47,111,208],[27,158,138],[224,163,43]];
const store = (window.__flakecvLabels = window.__flakecvLabels || {});

function loadImg(url){ return new Promise((ok,bad)=>{ const im=new Image(); im.onload=()=>ok(im); im.onerror=bad; im.src=url; }); }

export default function(component){
  const { data, parentElement, setTriggerValue } = component;
  if(!data || !data.image) return;
  const key = data.key, token = data.init_token;
  const old = parentElement.querySelector('.root');
  if(old && old.dataset.key===key && old.dataset.token===token) return;   // same frame: keep painting
  if(old) old.remove();
  if(store.__cleanup) { try{ store.__cleanup(); }catch(e){} }

  const root=document.createElement('div'); root.className='root';
  root.dataset.key=key; root.dataset.token=token;
  root.innerHTML = `
   <div class="bar">
     <button class="fl" aria-pressed="true"><span class="dot" style="background:var(--flake)"></span>flake (1)</button>
     <button class="bg" aria-pressed="false"><span class="dot" style="background:var(--back)"></span>background (2)</button>
     <button class="un" aria-pressed="false"><span class="dot" style="background:var(--uncert)"></span>uncertain (3)</button>
     <span class="sep"></span>
     <span>brush</span><input type="range" class="brush" min="2" max="200" value="24"><span class="val vb"></span>
     <span>overlay</span><input type="range" class="opac" min="0" max="100" value="40">
     <span class="sep"></span>
     <button class="tbrush" aria-pressed="true">brush (B)</button>
     <button class="tpoly" aria-pressed="false">polygon (P)</button>
     <button class="tlasso" aria-pressed="false">lasso (L)</button>
     <button class="done" hidden>fill polygon ✓</button>
     <span class="sep"></span>
     <button class="fillh">fill holes</button>
     <button class="undo">undo (Z)</button><button class="fit">fit (F)</button>
     <span class="sep"></span>
     <button class="save">Save label</button>
   </div>
   <div class="stage"><canvas></canvas></div>
   <div class="msg"></div>
   <div class="keys"><b>polygon</b>: click the corners, then click the first corner again, double-click, press Enter or "fill polygon ✓" to fill it with the active class · <b>lasso</b>: hold and draw round the area, release to fill · scroll = zoom · right-drag or space-drag = pan · [ ] brush size · H hide overlay · Esc cancels a polygon</div>`;
  parentElement.appendChild(root);
  const q=s=>root.querySelector(s), cv=q('canvas'), ctx=cv.getContext('2d'), stage=q('.stage');
  const msg=(t,bad)=>{ q('.msg').textContent=t; q('.msg').style.color=bad?'#D4452A':''; };

  let W=0,H=0,lab=null,base=null,oc=null,cls=FL,brush=24,opac=.4,zoom=1,ox=0,oy=0,undo=[],poly=null,hide=false;
  let tool='brush', lasso=null, mouse=null, userView=false;
  let drawing=false,panning=false,last=null,panStart=null,space=false,hover=false;
  const px_um = data.px_um || 0;

  function size(){ const r=stage.getBoundingClientRect(); cv.width=Math.max(50,r.width|0); cv.height=Math.max(50,r.height|0); draw(); }
  function fit(){ if(!W) return; userView=false; zoom=Math.min((cv.width-10)/W,(cv.height-10)/H); ox=oy=0; draw(); }
  function off(){ return [(cv.width-W*zoom)/2+ox,(cv.height-H*zoom)/2+oy]; }
  function toImg(e){ const r=cv.getBoundingClientRect(), [dx,dy]=off(); return [(e.clientX-r.left-dx)/zoom,(e.clientY-r.top-dy)/zoom]; }
  function over(x0,y0,x1,y1){ x0=Math.max(0,x0|0);y0=Math.max(0,y0|0);x1=Math.min(W,Math.ceil(x1));y1=Math.min(H,Math.ceil(y1));
    if(x1<=x0||y1<=y0) return; const w=x1-x0,h=y1-y0,c=oc.getContext('2d'),d=c.createImageData(w,h),p=d.data;
    for(let y=0;y<h;y++){ let si=(y+y0)*W+x0,di=y*w*4; for(let x=0;x<w;x++,si++,di+=4){ const k=lab[si],cc=COL[k];
      p[di]=cc[0];p[di+1]=cc[1];p[di+2]=cc[2];p[di+3]=k===BG?0:255; } }
    c.putImageData(d,x0,y0); }
  function draw(){ if(!W) return; ctx.clearRect(0,0,cv.width,cv.height); ctx.imageSmoothingEnabled=false;
    const [dx,dy]=off(); ctx.drawImage(base,dx,dy,W*zoom,H*zoom);
    if(!hide){ ctx.globalAlpha=opac; ctx.drawImage(oc,dx,dy,W*zoom,H*zoom); ctx.globalAlpha=1; }
    const path=(pts,close)=>{ ctx.beginPath(); pts.forEach((p,i)=>{const X=dx+p[0]*zoom,Y=dy+p[1]*zoom; i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});
      if(close) ctx.closePath(); ctx.stroke(); };
    ctx.strokeStyle='#ff5030'; ctx.lineWidth=1.8;
    if(poly&&poly.length){ ctx.setLineDash([5,4]); path(mouse?[...poly,mouse]:poly,false); ctx.setLineDash([]);
      ctx.fillStyle='#ff5030'; poly.forEach((p,i)=>{ ctx.beginPath(); ctx.arc(dx+p[0]*zoom,dy+p[1]*zoom,i?3:6,0,7); i?ctx.fill():ctx.stroke(); }); }
    if(lasso&&lasso.length>1) path(lasso,true); }
  function stats(){ let f=0,u=0; for(let i=0;i<lab.length;i++){ if(lab[i]===FL)f++; else if(lab[i]===UN)u++; }
    const n=lab.length; msg(`flake ${(100*f/n).toFixed(1)}% · uncertain ${(100*u/n).toFixed(1)}%`+(px_um?` · flake area ${(f*px_um*px_um).toFixed(0)} µm²`:'')); }
  function pushUndo(){ undo.push(lab.slice()); if(undo.length>20) undo.shift(); }
  function disc(cx,cy,r,v){ const y0=Math.max(0,Math.floor(cy-r)),y1=Math.min(H,Math.ceil(cy+r)+1),r2=r*r;
    for(let y=y0;y<y1;y++){ const dy=y+.5-cy, w=Math.sqrt(Math.max(0,r2-dy*dy));
      const a=Math.max(0,Math.round(cx-w)), b=Math.min(W,Math.round(cx+w)); if(b>a) lab.fill(v,y*W+a,y*W+b); } }
  function stroke(a,b,v){ const r=brush/2,d=Math.hypot(b[0]-a[0],b[1]-a[1]),n=Math.max(1,Math.ceil(d/(r*.5)));
    for(let i=0;i<=n;i++){ const t=i/n; disc(a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,r,v); }
    over(Math.min(a[0],b[0])-r-1,Math.min(a[1],b[1])-r-1,Math.max(a[0],b[0])+r+2,Math.max(a[1],b[1])+r+2); draw(); }
  function fillPoly(pts,v){ const t=document.createElement('canvas'); t.width=W; t.height=H; const c=t.getContext('2d');
    c.fillStyle='#fff'; c.beginPath(); pts.forEach((p,i)=>i?c.lineTo(p[0],p[1]):c.moveTo(p[0],p[1])); c.closePath(); c.fill();
    const d=c.getImageData(0,0,W,H).data; for(let i=0,j=3;i<W*H;i++,j+=4) if(d[j]>127) lab[i]=v; over(0,0,W,H); draw(); }
  function fillHoles(){ // background pixels not connected to the image border become flake
    const seen=new Uint8Array(W*H), st=[];
    const push=i=>{ if(!seen[i]&&lab[i]===BG){ seen[i]=1; st.push(i);} };
    for(let x=0;x<W;x++){ push(x); push((H-1)*W+x); } for(let y=0;y<H;y++){ push(y*W); push(y*W+W-1); }
    while(st.length){ const i=st.pop(),x=i%W; if(x>0)push(i-1); if(x<W-1)push(i+1); if(i>=W)push(i-W); if(i<W*(H-1))push(i+W); }
    pushUndo(); let n=0; for(let i=0;i<W*H;i++) if(lab[i]===BG&&!seen[i]){ lab[i]=FL; n++; }
    over(0,0,W,H); draw(); stats(); }
  function setTool(t){ tool=t; poly = t==='poly'? [] : null; lasso=null;
    [['.tbrush','brush'],['.tpoly','poly'],['.tlasso','lasso']].forEach(([s,v])=>q(s).setAttribute('aria-pressed',t===v));
    q('.done').hidden=true; draw();
    if(t==='poly') msg('polygon: click the corners; click the first corner again (or double-click / Enter) to fill');
    if(t==='lasso') msg('lasso: hold the mouse button and draw round the area; release to fill'); }
  function finishPoly(){ if(!poly||poly.length<3) return false; pushUndo(); fillPoly(poly,cls); poly=[]; mouse=null; q('.done').hidden=true; draw();
    stats(); store[key]={token,lab}; return true; }
  function pick(c){ cls=c; [['.fl',FL],['.bg',BG],['.un',UN]].forEach(([s,v])=>q(s).setAttribute('aria-pressed',c===v)); }
  function setBrush(v){ brush=Math.min(200,Math.max(2,Math.round(v))); q('.brush').value=brush;
    q('.vb').textContent=brush+' px'+(px_um?` (${(brush*px_um).toFixed(1)} µm)`:''); }

  q('.fl').onclick=()=>pick(FL); q('.bg').onclick=()=>pick(BG); q('.un').onclick=()=>pick(UN);
  q('.brush').oninput=e=>setBrush(+e.target.value);
  q('.opac').oninput=e=>{ opac=+e.target.value/100; draw(); };
  q('.undo').onclick=()=>{ if(!undo.length) return; lab.set(undo.pop()); over(0,0,W,H); draw(); stats(); };
  q('.fit').onclick=fit;
  q('.tbrush').onclick=()=>setTool('brush'); q('.tpoly').onclick=()=>setTool('poly'); q('.tlasso').onclick=()=>setTool('lasso');
  q('.done').onclick=finishPoly;
  cv.addEventListener('dblclick',e=>{ if(tool==='poly'){ e.preventDefault(); finishPoly(); } });
  q('.fillh').onclick=fillHoles;
  q('.save').onclick=()=>{ if(!W) return; const t=document.createElement('canvas'); t.width=W; t.height=H;
    const c=t.getContext('2d'), d=c.createImageData(W,H), p=d.data;
    for(let i=0,j=0;i<W*H;i++,j+=4){ const v=DISK[lab[i]]; p[j]=p[j+1]=p[j+2]=v; p[j+3]=255; }
    c.putImageData(d,0,0); setTriggerValue('saved', {key, png:t.toDataURL('image/png')}); msg('sending to the app…'); };

  cv.addEventListener('contextmenu',e=>e.preventDefault());
  cv.addEventListener('pointerenter',()=>hover=true); cv.addEventListener('pointerleave',()=>hover=false);
  cv.addEventListener('pointerdown',e=>{ if(!W) return; cv.setPointerCapture(e.pointerId);
    if(e.button===2||e.button===1||space){ panning=true; userView=true; panStart=[e.clientX-ox,e.clientY-oy]; return; }
    const p=toImg(e);
    if(tool==='poly'){ if(poly.length>=3 && Math.hypot(p[0]-poly[0][0],p[1]-poly[0][1])*zoom<10){ finishPoly(); return; }
      poly.push(p); q('.done').hidden=poly.length<3; draw(); return; }
    if(tool==='lasso'){ lasso=[p]; draw(); return; }
    pushUndo(); drawing=true; last=p; stroke(p,p,cls); });
  cv.addEventListener('pointermove',e=>{ if(panning){ ox=e.clientX-panStart[0]; oy=e.clientY-panStart[1]; draw(); return; }
    if(tool==='poly'&&poly&&poly.length){ mouse=toImg(e); draw(); return; }
    if(lasso){ lasso.push(toImg(e)); draw(); return; }
    if(!drawing) return; const p=toImg(e); stroke(last,p,cls); last=p; });
  const up=()=>{ if(drawing){ stats(); store[key]={token,lab}; }
    if(lasso){ const pts=lasso; lasso=null; if(pts.length>=3){ pushUndo(); fillPoly(pts,cls); stats(); store[key]={token,lab}; } else draw(); }
    drawing=false; panning=false; };
  cv.addEventListener('pointerup',up); cv.addEventListener('pointercancel',up);
  cv.addEventListener('wheel',e=>{ if(!W) return; e.preventDefault(); userView=true; const b=toImg(e);
    zoom=Math.min(80,Math.max(.03,zoom*Math.exp(-e.deltaY*.0016))); const a=toImg(e);
    ox+=(a[0]-b[0])*zoom; oy+=(a[1]-b[1])*zoom; draw(); },{passive:false});
  const kd=e=>{ if(!hover&&!(poly&&poly.length)) return; const tg=(e.composedPath()[0]||{}).tagName; if(tg==='INPUT'||tg==='TEXTAREA') return;
    if(e.code==='Space'){ space=true; e.preventDefault(); }
    if(e.key==='1')pick(FL); if(e.key==='2')pick(BG); if(e.key==='3')pick(UN);
    if(e.key==='[')setBrush(brush*.8); if(e.key===']')setBrush(brush*1.25);
    if(e.key==='z'||e.key==='Z')q('.undo').click(); if(e.key==='p'||e.key==='P')setTool('poly'); if(e.key==='l'||e.key==='L')setTool('lasso'); if(e.key==='b'||e.key==='B')setTool('brush');
    if(e.key==='f'||e.key==='F')fit(); if(e.key==='h'||e.key==='H'){ hide=!hide; draw(); }
    if(e.key==='Enter'&&poly&&poly.length>2){ e.preventDefault(); finishPoly(); }
    if(e.key==='Escape'&&poly){ poly=[]; mouse=null; q('.done').hidden=true; draw(); msg('polygon cancelled'); } };
  const ku=e=>{ if(e.code==='Space') space=false; };
  window.addEventListener('keydown',kd); window.addEventListener('keyup',ku);
  const ro=new ResizeObserver(()=>{ size(); if(W&&!userView) fit(); }); ro.observe(stage);
  store.__cleanup=()=>{ window.removeEventListener('keydown',kd); window.removeEventListener('keyup',ku); ro.disconnect(); };

  (async()=>{
    const im=await loadImg(data.image); W=im.width; H=im.height;
    base=document.createElement('canvas'); base.width=W; base.height=H; base.getContext('2d').drawImage(im,0,0);
    oc=document.createElement('canvas'); oc.width=W; oc.height=H;
    const kept=store[key];
    if(kept && kept.token===token && kept.lab.length===W*H){ lab=kept.lab; msg('restored your unsaved strokes'); }
    else { lab=new Uint8Array(W*H);
      if(data.init){ const li=await loadImg(data.init), t=document.createElement('canvas'); t.width=W; t.height=H;
        const c=t.getContext('2d'); c.drawImage(li,0,0); const d=c.getImageData(0,0,W,H).data;
        for(let i=0,j=0;i<W*H;i++,j+=4){ const v=d[j]; lab[i]= v>=200?UN:(v>60?FL:BG); } }
      store[key]={token,lab}; }
    setBrush(Math.max(4, px_um? 1.0/px_um : 24)); size(); fit(); over(0,0,W,H); draw(); stats();
  })();
}
"""


def component():
    """The mount callable (registered once per process)."""
    import streamlit as st
    return st.components.v2.component("flakecv_labeller", html=HTML, css=CSS, js=JS)
