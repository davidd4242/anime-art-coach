"""Structure analysis: line art, region segmentation / classification,
protection mask and the over-rendering (texture residual) map.

Images: float32 RGB 0..1 (H x W x 3).  Masks: bool or float32 0..1.
"""
import cv2
import numpy as np

from colorfix import to_lab, chroma, background_mask


def region_slices(labels, pad=0):
    """[(label, (slice_y, slice_x)), ...] bounding boxes of labels > 0."""
    from skimage.measure import regionprops
    H, W = labels.shape
    out = []
    for p in regionprops(labels):
        y0, x0, y1, x1 = p.bbox
        out.append((p.label, (slice(max(0, y0 - pad), min(H, y1 + pad)),
                              slice(max(0, x0 - pad), min(W, x1 + pad)))))
    return out


def _ellipse(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


# ============================================================== line art
def line_strength(img, lab=None, k=9):
    """Morphological black-hat on L*: how much darker a pixel is than the
    brightest surroundings within a k x k disc.  Thin dark strokes (narrower
    than k) respond strongly; wide dark areas (skirt, boots) do not, because
    closing cannot 'bridge' them.  Returned in L* units (0..100)."""
    if lab is None:
        lab = to_lab(img)
    L = lab[..., 0]
    return cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, _ellipse(k))


def line_mask(img, lab=None, strong=22.0, weak=11.0, min_area=24, max_core_l=58.0):
    """Dark thin strokes (line art), including coloured (dark brown / dark
    red) lines.

    Hysteresis on the black-hat response: pixels above `weak` are kept when
    their 8-connected component contains a pixel above `strong`; tiny
    components (texture specks) are dropped.  Thresholds are in L* units so
    a dark-red line on a red wing counts as long as it is clearly darker than
    its neighbourhood.  Components whose darkest pixel is still pale
    (L* > `max_core_l`) are soft fold shading painted on white cloth, not
    ink lines; they are left to the flattening step.
    """
    if lab is None:
        lab = to_lab(img)
    bh = line_strength(img, lab)
    weak_m = (bh > weak).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(weak_m, connectivity=8)
    has_strong = np.zeros(n, bool)
    has_strong[np.unique(cc[bh > strong])] = True
    has_strong[0] = False
    keep = has_strong & (stats[:, cv2.CC_STAT_AREA] >= min_area)
    # darkest L* per component
    core = np.full(n, 100.0, np.float32)
    np.minimum.at(core, cc.ravel(), lab[..., 0].ravel())
    keep &= core <= max_core_l
    return keep[cc]


# ============================================================== L0 smoothing
def l0_smooth(img, lam=0.02, kappa=2.0, beta_max=1e5, max_side=None):
    """L0 gradient minimisation (Xu et al. 2011), FFT solver in numpy.

    Produces a piecewise-constant 'structure' image: soft gradients collapse
    into flat steps, strong edges survive.  `max_side` optionally runs the
    solver on a downscaled copy (result is upscaled back) for speed.
    """
    src = img.astype(np.float32)
    H0, W0 = src.shape[:2]
    if max_side and max(H0, W0) > max_side:
        s = max_side / max(H0, W0)
        src = cv2.resize(src, (int(W0 * s), int(H0 * s)), interpolation=cv2.INTER_AREA)
    if src.ndim == 2:
        src = src[..., None]
    H, W, C = src.shape
    # reflect-pad so the FFT's periodic boundary does not bleed across edges
    p = 8
    S = np.pad(src, ((p, p), (p, p), (0, 0)), mode="reflect")
    Hp, Wp = S.shape[:2]
    # optical transfer functions of forward differences
    fx = np.zeros((Hp, Wp), np.float32); fx[0, 0] = -1; fx[0, -1] = 1
    fy = np.zeros((Hp, Wp), np.float32); fy[0, 0] = -1; fy[-1, 0] = 1
    Fx = np.fft.rfft2(fx); Fy = np.fft.rfft2(fy)
    denom_grad = (np.abs(Fx) ** 2 + np.abs(Fy) ** 2).astype(np.float32)[..., None]
    FI = np.fft.rfft2(S, axes=(0, 1))
    beta = 2.0 * lam
    while beta < beta_max:
        h = np.roll(S, -1, axis=1) - S
        v = np.roll(S, -1, axis=0) - S
        small = (h ** 2 + v ** 2).sum(2) < lam / beta
        h[small] = 0
        v[small] = 0
        # divergence-like term: D^T h + D^T v
        dh = np.roll(h, 1, axis=1) - h
        dv = np.roll(v, 1, axis=0) - v
        num = FI + beta * np.fft.rfft2(dh + dv, axes=(0, 1))
        S = np.fft.irfft2(num / (1.0 + beta * denom_grad), s=(Hp, Wp), axes=(0, 1))
        S = S.astype(np.float32)
        beta *= kappa
    S = S[p:-p, p:-p]
    if (H, W) != (H0, W0):
        S = cv2.resize(S, (W0, H0), interpolation=cv2.INTER_LINEAR)
        if S.ndim == 2:
            S = S[..., None]
    out = S if img.ndim == 3 else S[..., 0]
    return np.clip(out, 0, 1).astype(np.float32)


# ============================================================== segmentation
class _UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        p = self.p
        r = x
        while p[r] != r:
            r = p[r]
        while p[x] != r:  # path compression
            p[x], x = r, p[x]
        return r


def _adjacency(labels, lab=None, gap=3):
    """Unique 4-neighbour label pairs (a<b, both >0), their boundary length
    and, if `lab` is given, the mean lightness-down-weighted colour step
    measured across the boundary between pixels `gap` apart.  A soft
    gradient gives a tiny step even if the two sides' means differ a lot; a
    material edge gives the full step."""
    n1 = int(labels.max()) + 1
    keys, steps = [], []
    H, W = labels.shape
    h = gap // 2
    for axis in (1, 0):
        if axis == 1:
            a, b = labels[:, :-1], labels[:, 1:]
        else:
            a, b = labels[:-1, :], labels[1:, :]
        m = (a != b) & (a > 0) & (b > 0)
        lo = np.minimum(a[m], b[m]).astype(np.int64)
        hi = np.maximum(a[m], b[m]).astype(np.int64)
        keys.append(lo * n1 + hi)
        if lab is not None:
            ys, xs = np.nonzero(m)
            if axis == 1:
                x0 = np.clip(xs - h, 0, W - 1); x1 = np.clip(xs + 1 + h, 0, W - 1)
                d = lab[ys, x1] - lab[ys, x0]
            else:
                y0 = np.clip(ys - h, 0, H - 1); y1 = np.clip(ys + 1 + h, 0, H - 1)
                d = lab[y1, xs] - lab[y0, xs]
            steps.append(np.sqrt((0.5 * d[:, 0]) ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2))
    key = np.concatenate(keys)
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    step = None
    if lab is not None:
        step = np.bincount(inv, weights=np.concatenate(steps)) / cnt
    return uk // n1, uk % n1, cnt, step


def _region_means(lab, labels, n):
    flat = labels.ravel()
    area = np.bincount(flat, minlength=n).astype(np.float64)
    means = np.stack([np.bincount(flat, weights=lab[..., c].ravel(), minlength=n)
                      for c in range(3)], 1) / np.maximum(area, 1)[:, None]
    return means, area


def _wde(m1, m2, wl=0.5):
    """Colour distance with lightness down-weighted: materials differ mostly
    in hue/chroma, while shading inside a material differs mostly in L*."""
    d = m1 - m2
    return np.sqrt((wl * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)


def _split_bimodal(lab, labels, excl, min_px=2000, sep_abs=7.0, sep_rel=0.45,
                   min_frac=0.12):
    """Split a region in two when its pixels form two clearly separated
    clusters in the a*b* (hue/chroma) plane.

    Single linkage occasionally glues two pale materials together through an
    airbrushed fade (a hair tip dissolving into the forehead skin).  Shading
    inside one material moves mostly along L*, so a bimodal a*b* distribution
    is a good sign of two materials.  The separation must be large both in
    absolute ΔE and relative to the region's chroma, which keeps saturated
    gradients (red wing: light pink -> deep red) in one piece.
    """
    out = labels.copy()
    nxt = int(labels.max()) + 1
    for r, sl in region_slices(labels):
        lab_c = lab[sl]
        full = labels[sl] == r
        m = full & ~excl[sl]
        cnt = int(m.sum())
        if cnt < min_px:
            continue
        ab = lab_c[..., 1:]
        px = ab[m].astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
        _, lbl, cen = cv2.kmeans(px, 2, None, crit, 2, cv2.KMEANS_PP_CENTERS)
        sep = float(np.linalg.norm(cen[0] - cen[1]))
        frac = float(min(lbl.mean(), 1 - lbl.mean()))
        cmean = float(np.linalg.norm(px.mean(0)))
        if sep < sep_abs or sep < sep_rel * cmean or frac < min_frac:
            continue
        # smooth the assignment so the cut follows a clean contour
        d0 = np.linalg.norm(ab - cen[0], axis=2)
        d1 = np.linalg.norm(ab - cen[1], axis=2)
        diff = np.where(full, d1 - d0, 0).astype(np.float32)
        wgt = cv2.GaussianBlur(full.astype(np.float32), (0, 0), 2.0)
        diff = cv2.GaussianBlur(diff, (0, 0), 2.0) / np.maximum(wgt, 1e-3)
        side = full & (diff < 0)          # closer to centre 1
        sub = out[sl]
        sub[side] = nxt
        nxt += 1
    return out


def segment_regions(img, lines, n_segments=2500, compactness=10.0,
                    edge_de=4.0, link_de=10.0, max_drift=8.0, min_area=150,
                    bg=None, raw=None):
    """Flat-colour region segmentation.

    1. Background (near-white, connected to the border without crossing line
       art) gets label 0, line pixels are excluded.
    2. SLIC superpixels in Lab, restricted to the remaining pixels.
    3. Merge superpixels Kruskal-style (single linkage over the adjacency
       graph, softest boundary first).  An edge is mergeable when the colour
       step measured *across* the shared boundary (on the lightly blurred
       `raw` image) is < `edge_de` and the superpixel means differ by
       < `link_de` (lightness down-weighted).  The across-boundary step is what
       separates a soft gradient (tiny step, we want to merge and later
       flatten it) from a material/cel edge (big step).  A `max_drift` test
       on the merged components' means stops long chains from gluing two
       different materials together.
    4. Regions smaller than `min_area` are absorbed into their most similar
       neighbour unless they are strongly contrasting (eye catch-lights, gems,
       buckles), which stay separate.

    `img` should preferably be the de-textured (L0) image; `raw` is the
    original (white-balanced) image used for the across-boundary step
    (defaults to `img`).  Line pixels are
    finally given the label of the nearest region so every non-background
    pixel has a region (needed to paint under the line art); use `lines` to
    tell them apart.  Returns int32 labels, 0 = background.
    """
    from skimage.segmentation import slic, expand_labels
    from skimage.measure import label as label_cc

    lab = to_lab(img)
    H, W = lines.shape
    if bg is None:
        barrier = cv2.dilate(lines.astype(np.uint8), _ellipse(3)).astype(bool)
        bg = background_mask(img, lab, barrier=barrier)
    mask = ~lines & ~bg
    # SLIC over the whole frame (skimage's masked init is very slow), then
    # cut away background/lines and split superpixels that a line crossed
    # into separate connected pieces.
    fg_frac = max(float(mask.mean()), 0.05)
    sp = slic(img, n_segments=int(n_segments / fg_frac), compactness=compactness,
              start_label=1, convert2lab=True, max_num_iter=6)
    sp = (sp * mask).astype(np.int32)
    sp = label_cc(sp, connectivity=1, background=0).astype(np.int32)
    # crumbs between lines are re-attached to the nearest region at the end
    cnt = np.bincount(sp.ravel())
    crumbs = cnt < 30
    crumbs[0] = False
    sp[crumbs[sp]] = 0
    _, inv = np.unique(sp, return_inverse=True)
    sp = inv.reshape(H, W).astype(np.int32)
    n = int(sp.max()) + 1
    means, area = _region_means(lab, sp, n)

    # ---- Kruskal single-linkage merge with drift guard
    raw_lab = to_lab(cv2.GaussianBlur(raw if raw is not None else img, (0, 0), 1.0))
    a, b, _, step = _adjacency(sp, raw_lab)
    w = step
    w[_wde(means[a], means[b]) >= link_de] = np.inf
    order = np.argsort(w)
    uf = _UnionFind(n)
    csum = means * area[:, None]          # running colour sums per root
    carea = area.copy()
    for e in order:
        if w[e] >= edge_de:
            break
        ra, rb = uf.find(a[e]), uf.find(b[e])
        if ra == rb:
            continue
        ma = csum[ra] / carea[ra]
        mb = csum[rb] / carea[rb]
        if _wde(ma, mb) > max_drift:
            continue
        uf.p[rb] = ra
        csum[ra] += csum[rb]
        carea[ra] += carea[rb]
    roots = np.array([uf.find(i) for i in range(n)])
    roots[0] = 0
    _, relabel = np.unique(roots, return_inverse=True)
    labels = relabel.reshape(-1)[sp.ravel()].reshape(H, W).astype(np.int32)

    # ---- split regions that chained two different pastel materials
    labels = _split_bimodal(lab, labels, bg | lines)

    # ---- absorb tiny regions (repeat: absorbed ones may still be small)
    for _ in range(3):
        n = int(labels.max()) + 1
        means, area = _region_means(lab, labels, n)
        small = (area < min_area) & (np.arange(n) > 0) & (area > 0)
        if not small.any():
            break
        a, b, cnt, _ = _adjacency(labels)
        target = np.arange(n)
        best = np.full(n, np.inf)
        for x, y in ((a, b), (b, a)):
            d = _wde(means[x], means[y], wl=1.0)
            for xi, yi, di in zip(x, y, d):
                if small[xi] and di < best[xi] and area[yi] >= area[xi]:
                    best[xi] = di
                    target[xi] = yi
        # keep strongly contrasting little islands (catch-lights, gems ...)
        target[best > 30.0] = np.arange(n)[best > 30.0]
        if np.all(target == np.arange(n)):
            break
        # resolve chains a->b->c
        for _ in range(5):
            target = target[target]
        labels = target[labels]
        _, inv = np.unique(labels, return_inverse=True)
        labels = inv.reshape(H, W).astype(np.int32)

    # ---- give line pixels (and slivers missed by SLIC) the nearest region
    unl = (labels == 0) & ~bg
    if unl.any():
        tmp = labels.copy()
        tmp[unl] = 0
        filled = expand_labels(tmp, distance=12)
        labels = np.where(unl, filled, labels)
    labels[bg] = 0
    return labels.astype(np.int32)


# ============================================================== region stats
def region_stats(img, labels, lines=None):
    """Per-region statistics on the (structure) image, computed on non-line
    pixels: area, mean/median Lab, L* spread, centroid, bbox, hue, chroma."""
    lab = to_lab(img)
    stats = {}
    for r, sl in region_slices(labels):
        m = labels[sl] == r
        if lines is not None:
            mm = m & ~lines[sl]
            if mm.sum() >= 0.25 * m.sum():
                m = mm
        px = lab[sl][m]
        if len(px) == 0:
            continue
        ys, xs = np.nonzero(m)
        med = np.median(px, 0)
        p3, p97 = np.percentile(px[:, 0], [3, 97])
        stats[r] = {
            "gloss": float((p97 - med[0]) / (med[0] - p3 + 2.0)),
            "hi_tail": float(p97 - med[0]),
            "area": int((labels[sl] == r).sum()),
            "mean": px.mean(0), "median": med,
            "sdL": float(px[:, 0].std()),
            "cy": float(ys.mean() + sl[0].start), "cx": float(xs.mean() + sl[1].start),
            "bbox": (sl[0].start, sl[1].start, sl[0].stop, sl[1].stop),
            "chroma": float(np.hypot(med[1], med[2])),
            "hue": float(np.degrees(np.arctan2(med[2], med[1]))),
        }
    return stats


# ============================================================== classification
def classify_regions(img, labels, lines=None, stats=None):
    """Heuristic kind per region: background, skin, hair, eyes, metal, cloth,
    other.  Colour/position/area rules only (no learning):

    * skin  : light (L*>55), low-moderate chroma (5..28), hue 25..68 deg
              (peach/pink-orange).  Pale skin in this style sits at C~8, h~50.
    * hair  : light (L*>55), chroma >= 8, hue 68..105 deg (blond/yellow).
              Other hair colours fall through to cloth/other - acceptable.
    * eyes  : found relative to the face (largest skin region in the top
              third of the figure): small, saturated or very dark/bright
              regions in the upper part of the face box, grouped per eye; all
              non-skin/non-hair regions inside each eye box become eyes.
    * metal : near-neutral (C<7), mid lightness, small (< 0.4 % of figure).
    * cloth : any other region >= 0.2 % of the figure; smaller ones: other.
    """
    if stats is None:
        stats = region_stats(img, labels, lines)
    kinds = {0: "background"}
    fg_area = max(int((labels > 0).sum()), 1)
    ys, xs = np.nonzero(labels > 0)
    fy0, fy1 = (ys.min(), ys.max()) if len(ys) else (0, labels.shape[0])
    fh = max(fy1 - fy0, 1)
    for r, st in stats.items():
        L, C, h = st["median"][0], st["chroma"], st["hue"]
        frac = st["area"] / fg_area
        if L > 55 and 5.0 <= C <= 28 and 25 <= h <= 68:
            k = "skin"
        elif L > 55 and C >= 8 and 68 < h <= 105:
            k = "hair"
        elif C < 7 and 30 < L < 88 and frac < 0.004:
            k = "metal"
        elif frac >= 0.002:
            k = "cloth"
        else:
            k = "other"
        kinds[r] = k

    # ---- face & eyes
    face = None
    best = 0
    for r, st in stats.items():
        if kinds[r] == "skin" and st["cy"] < fy0 + 0.33 * fh and st["area"] > best:
            best, face = st["area"], r
    if face is not None:
        y0, x0, y1, x1 = stats[face]["bbox"]
        fw, fhh = x1 - x0, y1 - y0
        cx = 0.5 * (x0 + x1)
        cand = []
        for r, st in stats.items():
            if kinds[r] in ("skin", "hair") or r == face:
                continue
            L, C = st["median"][0], st["chroma"]
            if not (x0 + 0.08 * fw < st["cx"] < x1 - 0.08 * fw):
                continue
            if not (y0 < st["cy"] < y0 + 0.55 * fhh):
                continue
            if st["area"] > 0.08 * stats[face]["area"]:
                continue
            if C > 20 or L < 35:
                cand.append(r)
        for side in (-1, 1):
            rs = [r for r in cand if (stats[r]["cx"] - cx) * side > 0]
            if not rs:
                continue
            by0 = min(stats[r]["bbox"][0] for r in rs)
            bx0 = min(stats[r]["bbox"][1] for r in rs)
            by1 = max(stats[r]["bbox"][2] for r in rs)
            bx1 = max(stats[r]["bbox"][3] for r in rs)
            for r, st in stats.items():
                if kinds[r] in ("skin", "hair") or r == face:
                    continue
                if by0 <= st["cy"] <= by1 and bx0 <= st["cx"] <= bx1 and \
                        st["area"] < 0.08 * stats[face]["area"]:
                    kinds[r] = "eyes"
    return kinds


def group_materials(img, labels, kinds, stats=None, ab_tol=9.0, l_tol=32.0):
    """Group adjacent regions of compatible kind and similar hue/chroma
    (a*b* distance < `ab_tol`, L* difference < `l_tol`) into one 'material'
    that will share a single cel palette.  Highlight streaks and shadow
    pieces that segmentation cut out of a material (they are usually tagged
    'other') rejoin it here.

    Segmentation splits a material wherever a line or a hard shadow edge
    crosses it (wing bones, skirt pleats, folds).  Flattening each piece on
    its own would give every piece its own slightly different levels; a
    shared palette is what a cel painter would use.  Adjacency is taken on
    the line-filled label map, so pieces separated only by a line are
    neighbours.  Returns int32 array mapping label -> group id (0 = bg).
    """
    if stats is None:
        stats = region_stats(img, labels)
    n = int(labels.max()) + 1
    a, b, cnt, _ = _adjacency(labels)
    uf = _UnionFind(n)
    med = {r: st["median"] for r, st in stats.items()}
    # strongest-shared-boundary first, so big coherent pieces merge first
    for e in np.argsort(-cnt):
        ra_, rb_ = int(a[e]), int(b[e])
        if ra_ not in med or rb_ not in med:
            continue
        ka, kb = kinds.get(ra_), kinds.get(rb_)
        if "background" in (ka, kb) or "eyes" in (ka, kb):
            continue
        # skin and hair are identity materials: never mix them with anything
        # else (a chain of small pale crumbs could otherwise drag a white
        # sock into the leg's skin group)
        if ka != kb and ({ka, kb} & {"skin", "hair"}):
            continue
        # do not let a glossy piece (belt, boot) melt into matte fabric
        # (skirt) of the same brown: compare lightness-distribution skew
        ga, gb = stats[ra_], stats[rb_]
        if min(ga["area"], gb["area"]) > 400:
            sha = ga["gloss"] > 0.8 and ga["hi_tail"] > 12
            shb = gb["gloss"] > 0.8 and gb["hi_tail"] > 12
            if sha != shb and min(ga["gloss"], gb["gloss"]) < 0.55:
                continue
        ma, mb = med[ra_], med[rb_]
        if np.hypot(ma[1] - mb[1], ma[2] - mb[2]) < ab_tol and abs(ma[0] - mb[0]) < l_tol:
            x, y = uf.find(ra_), uf.find(rb_)
            if x != y:
                uf.p[y] = x
    # eyes: every eye region of one eye forms one group (eye = one palette)
    eye = [r for r, k in kinds.items() if k == "eyes"]
    for i in eye:
        for j in eye:
            if i < j and abs(stats[i]["cx"] - stats[j]["cx"]) < 0.06 * labels.shape[1]:
                x, y = uf.find(i), uf.find(j)
                if x != y:
                    uf.p[y] = x
    groups = np.array([uf.find(i) for i in range(n)], np.int32)
    groups[0] = 0
    return groups


# ============================================================== protection
def protect_mask(img, labels, kinds, stats=None, eyes=True, eye_level=0.35,
                 acc_frac=0.0015, acc_contrast=25.0, reflections=None):
    """Soft 0..1 mask of things that must survive the redraw.

    * Eyes: the iris colours may be flattened (budget 'eyes'), so the eye
      area gets a moderate `eye_level`; small bright catch-lights and the
      dark pupil core inside the eye get 1.0 so their crisp shapes survive.
    * Small accessories: regions smaller than `acc_frac` of the figure whose
      colour differs by > `acc_contrast` ΔE from their neighbours (gems,
      buckles, studs, the cross) get 1.0.
    """
    if stats is None:
        stats = region_stats(img, labels)
    lab = to_lab(img)
    H, W = labels.shape
    prot = np.zeros((H, W), np.float32)
    fg_area = max(int((labels > 0).sum()), 1)
    n = int(labels.max()) + 1
    a, b, cnt, _ = _adjacency(labels)
    nb = {}
    for x, y, c in zip(a, b, cnt):
        nb.setdefault(int(x), []).append((int(y), c))
        nb.setdefault(int(y), []).append((int(x), c))
    for r, st in stats.items():
        k = kinds.get(r)
        sl = (slice(st["bbox"][0], st["bbox"][2]), slice(st["bbox"][1], st["bbox"][3]))
        m = labels[sl] == r
        if k == "eyes" and eyes:
            sub = prot[sl]
            sub[m] = np.maximum(sub[m], eye_level)
            Lr = lab[sl][..., 0]
            med = st["median"][0]
            spark = m & ((Lr > max(med + 25, 85)) | (Lr < min(med - 25, 25)))
            sub[spark] = 1.0
        elif k not in ("background", "eyes") and st["area"] < acc_frac * fg_area:
            nbs = [(y, c) for y, c in nb.get(r, []) if y in stats]
            if not nbs:
                continue
            wsum = sum(c for _, c in nbs)
            nmean = sum(stats[y]["median"] * c for y, c in nbs) / wsum
            if np.linalg.norm(st["median"] - nmean) > acc_contrast:
                sub = prot[sl]
                sub[m] = 1.0
    # Reflection islands reassigned to their leather host have no hardware
    # edge. Remove their seeds before feathering to avoid a false bright ring.
    if reflections is not None:
        prot[reflections] = 0
    return cv2.GaussianBlur(prot, (0, 0), 1.2)


# ============================================================== over-render
def overrender_map(img, lines, structure=None, scale=6.0, floor=1.5):
    """Structure/texture residual: |img - L0(img)| as a ΔE-like magnitude,
    ignoring line art and true edges, blurred into a density, minus a noise
    `floor`, mapped to 0..1 with a soft saturation (ΔE `scale` -> 0.63).

    High values = soft gradients, airbrush, gloss, noise inside flat areas,
    i.e. the 'over-rendering' that makes the image look AI-painted.
    """
    if structure is None:
        structure = l0_smooth(img, lam=0.02)
    lab_i = to_lab(img)
    lab_s = to_lab(structure)
    d = lab_i - lab_s
    res = np.sqrt((0.7 * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
    # ignore line art and the immediate neighbourhood of structural edges
    gx = cv2.Sobel(lab_s[..., 0], cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(lab_s[..., 0], cv2.CV_32F, 0, 1)
    edge = np.hypot(gx, gy) > 40
    ign = cv2.dilate((lines | edge).astype(np.uint8), _ellipse(5)).astype(bool)
    res = np.where(ign, 0, res).astype(np.float32)
    w = cv2.GaussianBlur((~ign).astype(np.float32), (0, 0), 5)
    dens = cv2.GaussianBlur(res, (0, 0), 5) / np.maximum(w, 0.05)
    # subtract the compression-noise floor (flat webp/jpeg areas ~ΔE 1-1.5)
    out = 1.0 - np.exp(-np.maximum(dens - floor, 0) / scale)
    out[lines] = 0
    return out.astype(np.float32)


def group_kinds(groups, kinds, stats):
    """Kind of each material group = kind of its largest member region."""
    best = {}
    for r, st in stats.items():
        g = int(groups[r])
        if g == 0:
            continue
        if g not in best or st["area"] > best[g][0]:
            best[g] = (st["area"], kinds.get(r, "other"))
    out = {0: "background"}
    out.update({g: k for g, (_, k) in best.items()})
    return out


# ============================================================== materials
MATERIALS = ("background", "skin", "hair", "eyes", "cloth_light", "cloth_dark",
             "leather_glossy", "leather_strap", "metal", "gem", "membrane", "horn")
MATERIAL_COLORS = {  # debug_materials.png palette
    "background": (1.0, 1.0, 1.0), "skin": (1.0, 0.72, 0.55), "hair": (1.0, 0.9, 0.1),
    "eyes": (0.9, 0.0, 0.0), "cloth_light": (0.55, 0.85, 1.0), "cloth_dark": (0.1, 0.3, 0.85),
    "leather_glossy": (0.55, 0.25, 0.05), "metal": (0.55, 0.55, 0.6), "gem": (1.0, 0.0, 0.8),
    "membrane": (0.2, 0.8, 0.3), "horn": (0.76, 0.63, 0.5),
    "leather_strap": (0.47, 0.26, 0.36),
}


def texture_features(img, lines, sigma=4.0):
    """Per-pixel structure-tensor texture energy and orientation coherence
    of L*.  Fabric folds = many elongated, roughly parallel shading strokes
    -> high energy with high coherence; gloss = isolated blobs -> energy
    with low coherence; flat paint -> low energy."""
    L = cv2.GaussianBlur(to_lab(img)[..., 0], (0, 0), 1.0)
    ix = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3) / 8
    iy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3) / 8
    # line art would dominate the tensor: zero its gradients
    zone = cv2.dilate(lines.astype(np.uint8), _ellipse(5)).astype(bool)
    ix[zone] = 0
    iy[zone] = 0
    jxx = cv2.GaussianBlur(ix * ix, (0, 0), sigma)
    jyy = cv2.GaussianBlur(iy * iy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(ix * iy, (0, 0), sigma)
    tr = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (tr + 1e-3)
    return np.sqrt(tr), coh


def material_features(img, labels, lines, spec=None, over=None):
    """Generic per-region features used by classify_material (no models):
    base colour (median Lab, chroma, hue), area fraction, compactness
    (4*pi*area/perimeter^2 of the line-filled region), vertical/horizontal
    position in the figure box, texture energy + coherence (structure
    tensor), interior line density, specular blob density/coverage,
    over-render score and L* spread."""
    H, W = labels.shape
    lab = to_lab(img)
    energy, coh = texture_features(img, lines)
    fg = labels > 0
    ys, xs = np.nonzero(fg)
    fy0, fy1, fx0, fx1 = ys.min(), ys.max(), xs.min(), xs.max()
    fh, fw = max(fy1 - fy0, 1), max(fx1 - fx0, 1)
    fg_area = max(int(fg.sum()), 1)
    feats = {}
    for r, sl in region_slices(labels, pad=2):
        full = labels[sl] == r
        area = int(full.sum())
        if area < 20:
            continue
        inner = cv2.erode(full.astype(np.uint8), _ellipse(5)).astype(bool) & ~lines[sl]
        if inner.sum() < 10:
            inner = full & ~lines[sl]
        if inner.sum() < 5:
            inner = full
        px = lab[sl][inner]
        med = np.median(px, 0)
        cnts, _ = cv2.findContours(full.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        per = sum(cv2.arcLength(c, True) for c in cnts)
        e = energy[sl][inner]
        cw = coh[sl][inner]
        p3, p97 = np.percentile(px[:, 0], [3, 97])
        f = {
            "area_frac": area / fg_area,
            # gloss = bright tail longer than the dark tail (glints on a
            # darker base); fabric shading has the long tail on the dark side
            "gloss": float((p97 - med[0]) / (med[0] - p3 + 2.0)),
            "hi_tail": float(p97 - med[0]),
            "L": float(med[0]), "a": float(med[1]), "b": float(med[2]),
            "C": float(np.hypot(med[1], med[2])),
            "hue": float(np.degrees(np.arctan2(med[2], med[1]))),
            "sdL": float(px[:, 0].std()),
            "compact": float(4 * np.pi * area / max(per, 1) ** 2),
            "ypos": float((np.nonzero(full)[0].mean() + sl[0].start - fy0) / fh),
            "xoff": float(abs(np.nonzero(full)[1].mean() + sl[1].start - 0.5 * (fx0 + fx1)) / fw),
            "tex_energy": float(e.mean()),
            "tex_coherence": float((cw * e).sum() / max(e.sum(), 1e-6)),
            "line_density": float(lines[sl][full].mean()),
        }
        if spec is not None:
            sp = spec[sl] & full
            n, _, st, _ = cv2.connectedComponentsWithStats(sp.astype(np.uint8), connectivity=8)
            f["spec_blobs_per_10k"] = float((n - 1) * 1e4 / area)
            f["spec_cover"] = float(sp.sum() / area)
        if over is not None:
            f["overrender"] = float(over[sl][inner].mean())
        feats[r] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()}
    return feats


def classify_material(img, labels, lines, kinds=None, spec=None, over=None, feats=None):
    """Material of each region (redraw passes material groups).

    Identity materials come from classify_regions (`kinds`: skin, hair,
    eyes).  Everything else is decided from generic features, first match
    wins.  The key discriminator between glossy and matte materials is the
    *asymmetry of the lightness distribution*: glints sit on a darker base,
    so the bright tail is long (`gloss` = (p97-median)/(median-p3) high,
    `hi_tail` = p97-median large); fabric shading pushes pixels *darker*
    than the base (gloss < 0.7).  Position is not used for decisions.

      gem            small (<0.5 % of figure), saturated (C>25), glossy
                     (gloss>0.8 or sdL>14)
      metal          near-neutral (C<12), small (<1 %), not pale flat fabric
                     (L<90) and shiny/high-contrast (sdL>12 or gloss>0.6)
      membrane       saturated (C>30), mid-light (L>35), large (>0.5 %) -
                     a big strongly coloured non-skin surface (wings)
      leather_glossy darkish (L<60) with a long bright tail
                     (gloss>0.8 and hi_tail>12 L*) or many glint blobs
                     (>=4 per 10k px)
      cloth_light    pale (L>=68), low chroma (C<16)
      cloth_dark     everything else (fabric of any darker/coloured tone)
    """
    if feats is None:
        feats = material_features(img, labels, lines, spec, over)
    out = {0: "background"}
    for r, f in feats.items():
        k = (kinds or {}).get(r)
        if k in ("skin", "hair", "eyes", "background", "horn", "leather_strap"):
            out[r] = k
            continue
        L, C, area = f["L"], f["C"], f["area_frac"]
        gloss, tail, sd = f["gloss"], f["hi_tail"], f["sdL"]
        blobs = f.get("spec_blobs_per_10k", 0.0)
        if area < 0.005 and C > 25 and (gloss > 0.8 or sd > 14):
            m = "gem"
        elif C < 12 and area < 0.01 and L < 90 and (sd > 12 or gloss > 0.6):
            m = "metal"
        elif C > 30 and L > 35 and area > 0.005:
            m = "membrane"
        elif L < 60 and ((gloss > 0.8 and tail > 12) or blobs >= 4.0):
            m = "leather_glossy"
        elif L >= 68 and C < 16:
            m = "cloth_light"
        else:
            m = "cloth_dark"
        out[r] = m
    return out


# ============================================================== line tiers
TIER_NONE, TIER_OUTLINE, TIER_FACE, TIER_INTER, TIER_INTRA = 0, 1, 2, 3, 4
TIER_NAMES = {TIER_OUTLINE: "outline", TIER_FACE: "face", TIER_INTER: "inter-material",
              TIER_INTRA: "intra-material"}
TIER_COLORS = {TIER_OUTLINE: (0.0, 0.0, 0.0), TIER_FACE: (0.9, 0.0, 0.1),
               TIER_INTER: (0.0, 0.45, 1.0), TIER_INTRA: (1.0, 0.6, 0.0)}


def silhouette_mask(img, lines=None, lab=None):
    """Character vs background (True = character).

    Background = near-white, low-chroma pixels connected to the image border
    without crossing line art (so gaps between legs / arm and body / wing and
    body that open to the border are background, while white clothing that
    is closed by lines stays part of the figure).  Pin-holes (< 0.01 % of the
    image) in the figure are filled; the figure keeps only components bigger
    than 0.05 % of the image (stray specks are dropped).
    """
    if lab is None:
        lab = to_lab(img)
    if lines is None:
        lines = line_mask(img, lab)
    barrier = cv2.dilate(lines.astype(np.uint8), _ellipse(3)).astype(bool)
    bg = background_mask(img, lab, barrier=barrier)
    fig = ~bg
    H, W = fig.shape
    n, cc, st, _ = cv2.connectedComponentsWithStats(bg.astype(np.uint8), connectivity=4)
    # tiny background specks enclosed by the figure -> figure (holes)
    n2, cc2, st2, _ = cv2.connectedComponentsWithStats((~fig).astype(np.uint8), connectivity=4)
    small = st2[:, cv2.CC_STAT_AREA] < 1e-4 * H * W
    small[0] = False
    fig |= small[cc2]
    n3, cc3, st3, _ = cv2.connectedComponentsWithStats(fig.astype(np.uint8), connectivity=8)
    big = st3[:, cv2.CC_STAT_AREA] >= 5e-4 * H * W
    big[0] = False
    return big[cc3]


def outline_band(silhouette, width=None):
    """Pixels within `width` px of the background (default ~0.4 % of the
    longer image side, >= 3 px): where the silhouette's contour line lives."""
    H, W = silhouette.shape
    if width is None:
        width = max(3, int(round(0.004 * max(H, W))))
    bg = (~silhouette).astype(np.uint8)
    return cv2.dilate(bg, _ellipse(2 * width + 1)).astype(bool) & silhouette


def face_feature_mask(labels, kinds, stats, silhouette, pad=None):
    """Area whose lines are facial features (eyes, brows, lashes, nose,
    mouth): the convex hull of the face skin region (largest skin region in
    the upper third of the figure) grown by `pad`, plus the eye regions grown
    by 2*pad.  `pad` defaults to ~0.5 % of the image height."""
    H, W = labels.shape
    if pad is None:
        pad = max(3, int(round(0.005 * H)))
    out = np.zeros((H, W), np.uint8)
    ys = np.nonzero(silhouette.any(1))[0]
    if len(ys) == 0:
        return out.astype(bool)
    fy0, fy1 = ys.min(), ys.max()
    face, best = None, 0
    for r, st in stats.items():
        if kinds.get(r) == "skin" and st["cy"] < fy0 + 0.33 * (fy1 - fy0) and st["area"] > best:
            face, best = r, st["area"]
    if face is not None:
        pts = cv2.findNonZero((labels == face).astype(np.uint8))
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(out, hull, 1)
        out = cv2.dilate(out, _ellipse(2 * pad + 1))
    eyes = np.isin(labels, [r for r, k in kinds.items() if k == "eyes"]).astype(np.uint8)
    if eyes.any():
        out |= cv2.dilate(eyes, _ellipse(4 * pad + 1))
    return out.astype(bool)


def line_tiers(lines, silhouette, face_mask, gmap, band=None, win=9):
    """Classify line pixels into tiers:
    outline (within the silhouette contour band), face (facial features),
    inter-material (the non-line pixels around it belong to >= 2 material
    groups: skirt|skin, boot|sock) and intra-material (folds/pleats inside
    one material).  Returns uint8 tier map (0 = not a line)."""
    if band is None:
        band = outline_band(silhouette)
    tiers = np.zeros(lines.shape, np.uint8)
    big = np.int32(gmap.max() + 1)
    g = gmap.astype(np.int32)
    gmax = cv2.dilate(np.where(lines, 0, g).astype(np.float32), _ellipse(win))
    gmin = cv2.erode(np.where(lines | (g == 0), big, g).astype(np.float32), _ellipse(win))
    inter = (gmax != gmin) & (gmin < big) & (gmax > 0)
    tiers[lines] = TIER_INTRA
    tiers[lines & inter] = TIER_INTER
    tiers[lines & face_mask] = TIER_FACE
    tiers[lines & band] = TIER_OUTLINE
    return tiers


def tier_debug(img, tiers):
    vis = 0.35 * img + 0.65
    for t, c in TIER_COLORS.items():
        vis[tiers == t] = c
    return vis
