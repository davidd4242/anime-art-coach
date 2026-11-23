"""Diffuse light on a plain narrow band wrapped around an exposed limb."""
import cv2
import numpy as np

from colorfix import to_lab, from_lab
from highlights import apply_paint
from masks import region_slices


def band_samples(region, skin):
    """Find the long axis and sample two exposed skin banks along the strip."""
    y, x = np.nonzero(region)
    points = np.column_stack((x,y)).astype(np.float32)
    centre, axes = cv2.PCACompute(points, mean=None)
    axis, normal = axes
    offset = points - centre
    u = offset[:,0] * axis[0] + offset[:,1] * axis[1]
    v = offset[:,0] * normal[0] + offset[:,1] * normal[1]
    span = np.ptp(u)
    width = u.size / max(span, 1.)
    if width < 4 or not 3.9 < span / width < 22:
        return None
    # Ignore the cut ends. On both broad sides the nearest material should
    # be bare skin; intersecting straps, buckles and boot seams fail this test.
    marks = np.linspace(u.min() + .1 * span, u.max() - .1 * span, 9)
    exposed = []
    for mark in marks:
        strip = abs(u - mark) < max(1.3, .025 * span)
        if strip.sum() < width:
            return None
        lo, hi = np.percentile(v[strip], [4, 96])
        if not .50 * width < hi-lo < 1.6 * width:
            return None
        near = max(2.2, .26 * width)
        samples = centre + mark * axis + np.array([lo-near,hi+near])[:,None] * normal
        sx, sy = np.rint(samples).astype(int).T
        if np.any((sx < 0) | (sy < 0) | (sx >= skin.shape[1]) | (sy >= skin.shape[0])):
            return None
        exposed.append(skin[sy,sx])
    if np.min(np.mean(exposed,axis=0)) < .75:
        return None
    return y, x, 2 * (u-u.min()) / span-1, width


def soften_limb_bands(stack, groups, materials, protect, strength):
    """Keep the broad cylindrical turn and dark contact edges; suppress glints."""
    candidates = [g for g,m in materials.items() if m == 'leather_glossy']
    if not candidates:
        return
    ids = np.where(np.isin(groups,candidates),groups,0)
    skin = np.isin(groups,[g for g,m in materials.items() if m == 'skin'])
    for g,sl in region_slices(ids,pad=max(8,round(.015 * min(ids.shape)))):
        count, pieces, areas, _ = cv2.connectedComponentsWithStats(np.uint8(ids[sl]==g))
        for i in range(1,count):
            if areas[i,cv2.CC_STAT_AREA] < 95:
                continue
            part = pieces==i
            sample = band_samples(part,skin[sl])
            if sample is None:
                continue
            y, x, u, width = sample
            distance = cv2.distanceTransform(part.astype(np.uint8),cv2.DIST_L2,5)
            flat = stack.flat[sl]
            diffuse = flat * stack.shadow[sl]
            under = diffuse + (1-diffuse)*stack.highlight[sl]
            ink = stack.lines[sl]
            seen = under + ink[...,3:]*(ink[...,:3]-under)
            lab = to_lab(seen)[y,x]
            interior = distance[y,x] > max(1.5,.14*width)
            # Robust cross-sections estimate one broad roll in the leather.
            # A sharp vertical specular streak contributes to few sections.
            knots = np.linspace(-.88, .88, 11)
            positions, medians = [], []
            for knot in knots:
                bank = interior & (abs(u-knot) < .095)
                if bank.sum() >= 3:
                    positions.append(np.median(u[bank]))
                    medians.append(np.median(lab[bank],axis=0))
            if len(positions) < 7:
                continue
            coeff = np.polynomial.polynomial.polyfit(positions,medians,2)
            roll = coeff[0] + u[:,None]*(coeff[1] + u[:,None]*coeff[2])
            # Saturated metal, intentional patterned strips or jagged tonal
            # sections provide insufficient evidence for a plain fabric band.
            if np.median(abs(lab[interior,0]-roll[interior,0])) > 5:
                continue
            pivot = np.median(lab[interior,0])
            ceiling = roll[:,0] - .32*np.maximum(roll[:,0]-pivot,0)
            excess = np.maximum(lab[:,0]-ceiling,0)
            edge = np.clip((distance[y,x]-.15)/1.4,0,1)
            amount = strength * (1-protect[sl][y,x]) * edge
            lab[:,0] -= amount * .97 * excess
            lab[:,1:] += (amount * np.clip(excess/9,0,1))[:,None]*(roll[:,1:]-lab[:,1:])
            target = from_lab(lab[None,:,:])[0]
            # Preserve the existing alpha and the separation between ink and
            # paint. In a glint trapped in both layers, darken both equally.
            ratio = np.minimum(target/np.maximum(seen[y,x],1e-6),1)
            base = under.copy()
            visible = seen.copy()
            base[y,x] *= ratio
            visible[y,x] *= ratio
            shadow, high, out_ink = apply_paint(flat,base,visible,ink[...,3],np.minimum(flat,base))
            edit = np.zeros_like(part)
            edit[y,x] = (amount*excess > .025)
            stack.shadow[sl][edit] = shadow[edit]
            stack.highlight[sl][edit] = high[edit]
            stack.lines[sl][edit] = out_ink[edit]
