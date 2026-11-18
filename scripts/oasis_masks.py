"""
oasis_masks.py - OASIS-1 -> SynthSeg -> region masks, in three steps.

Run this in your normal Python environment (NOT the SynthSeg one).
Needs: numpy, nibabel, matplotlib, scipy; pandas + openpyxl only for --demographics / --qc.

  1. extract  Pull only the atlas-registered, brain-masked images (*_t88_masked_gfc)
              out of the OASIS disc tarballs and save them as NIfTI (.nii.gz),
              which SynthSeg needs (it cannot read Analyze .img/.hdr).
  2. preview  PNG of three views per subject, with orientation letters, and the
              hippocampus overlaid in red once SynthSeg has run.
  3. masks    Binary masks per region + a volume table + sanity checks.

The converted .nii.gz files are the images for everything downstream (ViT included),
so the masks and the images always share one voxel grid.
"""

import argparse
import glob
import json
import os
import re
import sys
import tarfile

import numpy as np
import nibabel as nib

# FreeSurfer label values used by SynthSeg (cortical ones exist only with --parc)
ROIS = {
    "hippocampus": [17, 53],
    "amygdala": [18, 54],
    "entorhinal_parahippocampal": [1006, 2006, 1016, 2016],
    # primary region: medial temporal lobe = all three above
    "mtl": [17, 53, 18, 54, 1006, 2006, 1016, 2016],
    # lateral + inferior lateral ventricles
    "ventricles": [4, 43, 5, 44],
}
AXES = {"x": 0, "y": 1, "z": 2}
IMG_RE = re.compile(r"(OAS1_\d{4}_MR\d)_mpr.*_t88_masked_gfc\.(img|hdr)$")
SID_RE = re.compile(r"OAS1_\d{4}_MR\d")


# --------------------------------------------------------------------------- helpers
def sid_of(path):
    m = SID_RE.search(os.path.basename(path))
    return m.group(0) if m else None


def list_images(img_dir):
    return sorted(p for p in glob.glob(os.path.join(img_dir, "OAS1_*.nii.gz"))
                  if "_synthseg" not in p)


def find_seg(seg_dir, sid):
    hits = sorted(glob.glob(os.path.join(seg_dir, sid + "*synthseg*.nii*")))
    return hits[0] if hits else None


def seg_on_image_grid(seg_img, ref_img):
    """Return seg data on the image's voxel grid (resampling only if needed)."""
    same = (seg_img.shape[:3] == ref_img.shape[:3]
            and np.allclose(seg_img.affine, ref_img.affine, atol=1e-3))
    if same:
        return np.asanyarray(seg_img.dataobj).astype(np.int32), False
    from nibabel.processing import resample_from_to
    res = resample_from_to(seg_img, (ref_img.shape[:3], ref_img.affine), order=0)
    return np.asanyarray(res.dataobj).astype(np.int32), True


def analyze_to_nifti(hdr_path, out_path, flips):
    img = nib.load(hdr_path)
    data = np.squeeze(np.asanyarray(img.dataobj))  # OASIS files are 176x208x176x1
    if data.ndim != 3:
        raise ValueError("unexpected shape %s in %s" % (data.shape, hdr_path))
    for ax in flips:
        data = np.flip(data, axis=AXES[ax])
    data = np.ascontiguousarray(data)
    data = data.astype(np.int16 if np.issubdtype(data.dtype, np.integer) else np.float32)
    nib.save(nib.Nifti1Image(data, img.affine), out_path)


# --------------------------------------------------------------------------- 1. extract
def cmd_extract(a):
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "_tmp_analyze")
    os.makedirs(tmp, exist_ok=True)
    tars = sorted(glob.glob(os.path.join(a.tars, "oasis_cross-sectional_disc*.tar.gz")))
    if not tars:
        sys.exit("No oasis_cross-sectional_disc*.tar.gz files in %s" % a.tars)

    log_path = os.path.join(a.out, "conversion_log.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    if log.get("flips") not in (None, a.flip):
        sys.exit("Earlier files were converted with flips: %s. Delete the output folder "
                 "and rerun, so every subject gets the same orientation." % (log["flips"] or "none"))
    log["flips"] = a.flip
    log.setdefault("sources", {})

    done = 0
    for tar_path in tars:
        print("Reading", os.path.basename(tar_path), flush=True)
        pending, seen_names, found_here = {}, [], 0
        with tarfile.open(tar_path, "r|gz") as tf:  # streaming: one pass, no full unpack
            for member in tf:
                if len(seen_names) < 15:
                    seen_names.append(member.name)
                m = IMG_RE.search(os.path.basename(member.name))
                if not (member.isfile() and m):
                    continue
                found_here += 1
                sid, ext = m.group(1), m.group(2)
                out_path = os.path.join(a.out, sid + ".nii.gz")
                if os.path.exists(out_path):
                    continue
                with open(os.path.join(tmp, sid + "." + ext), "wb") as f:
                    f.write(tf.extractfile(member).read())
                pending.setdefault(sid, set()).add(ext)
                if pending[sid] == {"img", "hdr"}:
                    analyze_to_nifti(os.path.join(tmp, sid + ".hdr"), out_path, a.flip)
                    for e in ("img", "hdr"):
                        os.remove(os.path.join(tmp, sid + "." + e))
                    log["sources"][sid] = os.path.basename(tar_path)
                    done += 1
                    print("  ", sid, flush=True)
                    if a.limit and done >= a.limit:
                        break
        if found_here == 0:
            print("  No *_t88_masked_gfc images found. First entries in this archive:")
            for n in seen_names:
                print("    ", n)
        json.dump(log, open(log_path, "w"), indent=1)
        if a.limit and done >= a.limit:
            break
    try:
        os.rmdir(tmp)
    except OSError:
        pass
    print("Converted %d new image(s). Orientation flips used: %s" % (done, a.flip or "none"))


# --------------------------------------------------------------------------- 2. preview
def cmd_preview(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    red = ListedColormap(["red"])

    os.makedirs(a.out, exist_ok=True)
    paths = list_images(a.img)
    if a.ids:
        paths = [p for p in paths if sid_of(p) in a.ids]
    if not paths:
        sys.exit("No matching images in %s" % a.img)

    for p in paths:
        sid = sid_of(p)
        img = nib.load(p)
        codes = "".join(nib.aff2axcodes(img.affine))
        seg = None
        seg_path = find_seg(a.seg, sid) if a.seg else None
        if seg_path:
            seg, _ = seg_on_image_grid(nib.load(seg_path), img)
            seg = nib.Nifti1Image(seg.astype(np.int16), img.affine)

        # reorient to RAS for display only: x = left->right, y = back->front, z = down->up
        can = nib.as_closest_canonical(img)
        vol = np.asanyarray(can.dataobj).astype(np.float32)
        hip = None
        if seg is not None:
            s = np.asanyarray(nib.as_closest_canonical(seg).dataobj)
            hip = np.isin(s, ROIS["hippocampus"])
        if hip is not None and hip.any():
            left = s == 17
            cx, cy, cz = [int(round(c)) for c in np.argwhere(left if left.any() else hip).mean(0)]
        else:
            cx, cy, cz = [int(round(c)) for c in np.argwhere(vol > 0).mean(0)]

        views = [("Sagittal", vol[cx, :, :], None if hip is None else hip[cx, :, :], ("P", "A", "I", "S")),
                 ("Coronal", vol[:, cy, :], None if hip is None else hip[:, cy, :], ("L", "R", "I", "S")),
                 ("Axial", vol[:, :, cz], None if hip is None else hip[:, :, cz], ("L", "R", "P", "A"))]
        fig, axs = plt.subplots(1, 3, figsize=(13, 4.8))
        vmax = np.percentile(vol[vol > 0], 99.5) if (vol > 0).any() else 1
        for ax, (name, sl, hs, (lft, rgt, bot, top)) in zip(axs, views):
            ax.imshow(sl.T, cmap="gray", origin="lower", vmin=0, vmax=vmax)
            if hs is not None:
                ax.imshow(np.ma.masked_where(~hs.T, hs.T), cmap=red, origin="lower", alpha=0.5)
            ax.set_title(name)
            ax.axis("off")
            kw = dict(transform=ax.transAxes, color="yellow", fontsize=14, fontweight="bold",
                      ha="center", va="center")
            ax.text(0.03, 0.5, lft, **kw)
            ax.text(0.97, 0.5, rgt, **kw)
            ax.text(0.5, 0.03, bot, **kw)
            ax.text(0.5, 0.97, top, **kw)
        tag = "hippocampus in red" if hip is not None else "no segmentation yet"
        fig.suptitle("%s   header orientation: %s   (%s)" % (sid, codes, tag))
        fig.tight_layout()
        out = os.path.join(a.out, sid + ".png")
        fig.savefig(out, dpi=90)
        plt.close(fig)
        print("wrote", out)


# --------------------------------------------------------------------------- 3. masks
def read_qc(path):
    import pandas as pd
    df = pd.read_csv(path)
    num = df.select_dtypes("number")
    out = {}
    for i, row in df.iterrows():
        sid = sid_of(str(row.iloc[0]))
        if sid and not num.empty:
            out[sid] = float(num.loc[i].min())
    return out


def robust_outliers(values, k=4.0):
    v = np.asarray(values, float)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 or 1.0
    return np.abs(v - med) / mad > k


def cmd_masks(a):
    import csv
    for roi in ROIS:
        os.makedirs(os.path.join(a.out, roi), exist_ok=True)
    qc = read_qc(a.qc) if a.qc else {}

    rows, missing, warned_parc = [], [], False
    for p in list_images(a.img):
        sid = sid_of(p)
        seg_path = find_seg(a.seg, sid)
        if not seg_path:
            missing.append(sid)
            continue
        img = nib.load(p)
        seg, resampled = seg_on_image_grid(nib.load(seg_path), img)
        vox = float(np.prod(img.header.get_zooms()[:3]))
        row = {"sid": sid, "resampled": int(resampled),
               "hippo_L": int((seg == 17).sum() * vox), "hippo_R": int((seg == 53).sum() * vox)}
        for roi, labels in ROIS.items():
            m = np.isin(seg, labels)
            row[roi] = int(m.sum() * vox)
            if roi == "entorhinal_parahippocampal" and not m.any():
                if not warned_parc:
                    print("NOTE: no cortical parcellation labels found; was SynthSeg run without --parc?")
                    warned_parc = True
                continue
            nib.save(nib.Nifti1Image(m.astype(np.uint8), img.affine),
                     os.path.join(a.out, roi, "%s_%s.nii.gz" % (sid, roi)))
        lr = (row["hippo_L"], row["hippo_R"])
        row["hippo_asym"] = round(abs(lr[0] - lr[1]) / max(1, np.mean(lr)), 3)
        row["qc_min"] = qc.get(sid, "")
        rows.append(row)

    if not rows:
        sys.exit("No SynthSeg outputs matched the images. Check --seg.")

    flags = robust_outliers([r["hippocampus"] for r in rows])
    for r, f in zip(rows, flags):
        why = []
        if r["hippocampus"] == 0:
            why.append("no hippocampus")
        if f:
            why.append("volume outlier")
        if r["hippo_asym"] > 0.3:
            why.append("L/R asymmetry")
        if r["qc_min"] != "" and r["qc_min"] < a.qc_threshold:
            why.append("low QC score")
        r["flag"] = "; ".join(why)

    table = os.path.join(a.out, "roi_volumes.csv")
    with open(table, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    flagged = [r for r in rows if r["flag"]]
    print("Masks for %d sessions -> %s" % (len(rows), a.out))
    print("Volumes are in atlas space (already head-size normalised), so compare them, don't quote them.")
    if missing:
        print("No segmentation yet for %d session(s), e.g. %s" % (len(missing), ", ".join(missing[:5])))
    print("Flagged for a look: %d" % len(flagged))
    for r in flagged[:30]:
        print("   %s  %s" % (r["sid"], r["flag"]))

    if a.demographics:
        sanity_checks(rows, a.demographics)


def sanity_checks(rows, demo_path):
    import pandas as pd
    demo = pd.read_excel(demo_path) if demo_path.lower().endswith((".xls", ".xlsx")) else pd.read_csv(demo_path)
    vols = pd.DataFrame(rows)
    df = vols.merge(demo, left_on="sid", right_on="ID", how="left")

    old = df[(df["Age"] >= 60) & df["CDR"].notna()]
    nd, dem = old[old["CDR"] == 0]["hippocampus"], old[old["CDR"] > 0]["hippocampus"]
    print("\nCheck 1 - age 60+: mean hippocampus, non-demented %.0f (n=%d) vs dementia %.0f (n=%d)"
          % (nd.mean(), len(nd), dem.mean(), len(dem)))
    print("   Expect the dementia group clearly lower (roughly 10-25%%). Got %.1f%% lower."
          % (100 * (1 - dem.mean() / nd.mean())))

    # under-60s have no CDR in OASIS-1; they are all non-demented
    healthy = df[(df["CDR"].fillna(0) == 0) & df["Age"].notna()]
    r = np.corrcoef(healthy["Age"], healthy["hippocampus"])[0, 1]
    print("Check 2 - non-demented: correlation of hippocampus with age r = %.2f (expect negative)" % r)

    vols["subject"] = vols["sid"].str[:9]
    pairs = vols.groupby("subject").filter(lambda g: len(g) > 1)
    if len(pairs):
        diffs = [abs(g["hippocampus"].iloc[0] - g["hippocampus"].iloc[1]) / g["hippocampus"].mean()
                 for _, g in pairs.groupby("subject")]
        print("Check 3 - repeat scans: %d subjects, mean hippocampus difference %.1f%% (expect a few %%)"
              % (len(diffs), 100 * np.mean(diffs)))


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="tarballs -> .nii.gz")
    e.add_argument("--tars", required=True, help="folder with oasis_cross-sectional_disc*.tar.gz")
    e.add_argument("--out", required=True, help="output folder for .nii.gz images")
    e.add_argument("--flip", nargs="*", default=[], choices=list(AXES),
                   help="axes to flip if the preview shows the brain backwards/upside down")
    e.add_argument("--limit", type=int, default=0, help="stop after N new images (0 = all)")

    p = sub.add_parser("preview", help="orientation / overlay PNGs")
    p.add_argument("--img", required=True)
    p.add_argument("--seg", default=None, help="SynthSeg output folder (optional)")
    p.add_argument("--out", required=True)
    p.add_argument("--ids", nargs="*", default=None, help="e.g. OAS1_0001_MR1 (default: all)")

    m = sub.add_parser("masks", help="SynthSeg output -> region masks + checks")
    m.add_argument("--img", required=True)
    m.add_argument("--seg", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--qc", default=None, help="SynthSeg --qc CSV (optional)")
    m.add_argument("--qc-threshold", type=float, default=0.65)
    m.add_argument("--demographics", default=None, help="OASIS-1 demographics .xlsx/.csv (optional)")

    a = ap.parse_args()
    {"extract": cmd_extract, "preview": cmd_preview, "masks": cmd_masks}[a.cmd](a)


if __name__ == "__main__":
    main()
