"""Material analysis on the original painting."""
import cv2
import numpy as np
from colorfix import to_lab, chroma
from masks import region_slices, _ellipse

def detect_specular(img, labels, kinds, lines=None, skip=("background", "eyes"),
                    loose=("leather_glossy", "metal", "gem")):
    """Specular / airbrushed highlight blobs.

    Per region (redraw passes *material groups* here, so streaks that
    segmentation cut out as separate pieces are judged against the whole
    material): pixels clearly brighter than the region,
        L* > median + max(10, 0.25 (100 - median), 1.0 sd),
    not more saturated than the region, forming components of
    8 px .. 30 % of the region.  The 2-px anti-aliased fringe against the
    background and line art is ignored (it is lighter but not a highlight).
    Regions whose kind is in `loose` (glossy materials, once known) use a
    looser threshold, median + max(7, 0.7 sd), so the whole soft glint is
    caught, not just its peak.
    """
    lab = to_lab(cv2.GaussianBlur(img, (0, 0), 1.0))
    H, W = labels.shape
    spec = np.zeros((H, W), bool)
    for r, sl in region_slices(labels, pad=3):
        if kinds.get(r) in skip:
            continue
        full = labels[sl] == r
        m = cv2.erode(full.astype(np.uint8), _ellipse(5)).astype(bool)
        if lines is not None:
            m &= ~cv2.dilate(lines[sl].astype(np.uint8), _ellipse(3)).astype(bool)
        area = int(full.sum())
        if area < 150 or m.sum() < 50:
            continue
        L = lab[sl][..., 0]
        C = chroma(lab[sl])
        medL = float(np.median(L[m]))
        sdL = float(L[m].std())
        medC = float(np.median(C[m]))
        if kinds.get(r) in loose:
            thr = medL + max(7.0, 0.7 * sdL)
            cand = m & (L > thr) & (C < medC + 6.0)
        else:
            thr = medL + max(10.0, 0.25 * (100 - medL), 1.0 * sdL)
            cand = m & (L > thr) & (C < medC + 3.0)
        if not cand.any():
            continue
        nc, cc, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
        ok = (st[:, cv2.CC_STAT_AREA] >= 8) & (st[:, cv2.CC_STAT_AREA] <= 0.3 * area)
        ok[0] = False
        sub = spec[sl]
        sub |= ok[cc]
    return spec


def blush_strength(lab, base):
    """Pinkness relative to the skin base colour: extra a* (red) that is not
    explained by shading (shadows are both darker and redder)."""
    dL = lab[..., 0] - base[0]
    da = lab[..., 1] - base[1]
    db = lab[..., 2] - base[2]
    return da - 0.6 * np.maximum(0, -dL) - 0.5 * np.maximum(0, db - 0.5 * da)


def detect_blush(img, labels, kinds, lines=None, thr=3.5):
    """Soft pinkish patches inside skin regions (cheek blush, knee/elbow
    flush).  Components between 60 px and 25 % of the skin region."""
    lab = to_lab(cv2.GaussianBlur(img, (0, 0), 2.0))
    H, W = labels.shape
    out = np.zeros((H, W), bool)
    for r, sl in region_slices(labels):
        if kinds.get(r) != "skin":
            continue
        m = labels[sl] == r
        if lines is not None:
            m &= ~lines[sl]
        area = int(m.sum())
        if area < 500:
            continue
        base = np.median(to_lab(img[sl])[m], 0)
        s = blush_strength(lab[sl], base)
        cand = m & (s > thr)
        nc, cc, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
        ok = (st[:, cv2.CC_STAT_AREA] >= 60) & (st[:, cv2.CC_STAT_AREA] <= 0.25 * area)
        ok[0] = False
        sub = out[sl]
        sub |= ok[cc]
    return out


