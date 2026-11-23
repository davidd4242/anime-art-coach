"""Straight-alpha ink over screen highlights over multiply shadows, in display RGB."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from colorfix import save_image, to_lab
from masks import region_slices, _ellipse
from pigment import paired_banks


@dataclass
class Stack:
    flat: np.ndarray
    shadow: np.ndarray        # white is the identity multiplier
    highlight: np.ndarray     # black is the identity screen colour
    lines: np.ndarray         # straight RGBA, including antialias coverage

    def composite(self):
        diffuse = self.flat * self.shadow
        paint = diffuse + (1 - diffuse) * self.highlight
        alpha = self.lines[..., 3:]
        return paint * (1 - alpha) + self.lines[..., :3] * alpha

    def write(self, directory, source_alpha=None, hidden_rgb=None):
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)
        for name in ('flat', 'shadow', 'highlight', 'lines'):
            pixels = getattr(self, name)
            if name == 'flat' and source_alpha is not None:
                # Opacity belongs to the entire clipped layer group. Store it
                # once, on the base plate, so partial coverage never squares.
                if hidden_rgb is not None:
                    pixels = np.where((source_alpha == 0)[..., None], hidden_rgb, pixels)
                pixels = np.dstack((pixels, source_alpha.astype(np.float32) / 255))
            save_image(directory / ('layer_' + name + '.png'), pixels)


def lift_ink(img, lines, confidence):
    """Estimate paint under ink. Solve a feasible RGB ink for the estimated coverage."""
    zone = cv2.dilate(lines.astype(np.uint8), _ellipse(3))
    under = cv2.inpaint(np.uint8(img * 255 + 0.5), zone, 3, cv2.INPAINT_NS)
    under = np.where(zone[..., None] > 0, under.astype(np.float32) / 255, img)
    under = img + confidence[..., None] * (under - img)
    delta = img - under
    bound = np.where(delta < 0, -delta / np.maximum(under, 1e-6),
                     delta / np.maximum(1 - under, 1e-6)).max(2)
    alpha = np.maximum(bound, lines.astype(np.float32)*confidence).clip(0, 1)
    ink = np.clip(under + delta / np.maximum(alpha[..., None], 1e-6), 0, 1)
    ink[alpha == 0] = 0
    return under, np.dstack((ink, alpha))


def decompose(img, lines, groups, materials, intra):
    # A dark run joined to a pale fold used to classify the entire fold as ink.
    # Assign its translucent, low-pigment tail back to the paint layer.
    light_cloth = np.isin(groups, [g for g,m in materials.items() if m == 'cloth_light'])
    soft_fold = light_cloth & (cv2.dilate(intra.astype(np.uint8), _ellipse(3)) > 0)
    confidence = np.ones(groups.shape, np.float32)
    confidence[soft_fold] = np.clip((72-to_lab(img)[...,0][soft_fold])/18, 0, 1)
    under, ink = lift_ink(img, lines, confidence)
    flat = under.copy()  # background is its own unedited plate
    value = cv2.cvtColor(under, cv2.COLOR_RGB2GRAY)
    for g, sl in region_slices(groups, pad=2):
        piece = groups[sl] == g
        interior = cv2.erode(piece.astype(np.uint8), _ellipse(5)).astype(bool)
        interior &= ~lines[sl]
        if not interior.any():
            interior = piece
        v = value[sl][interior]
        mat = materials.get(g)
        # Diffuse material colours come from their lit mids. For polished
        # surfaces sample the body of the histogram below the reflection tail.
        quantiles = (40, 59) if mat in ('metal', 'gem', 'leather_glossy', 'leather_strap') else (74, 90)
        lo, hi = np.percentile(v, quantiles)
        sample = under[sl][interior]
        band = (v >= lo) & (v <= hi)
        flat[sl][piece] = np.median(sample[band] if band.any() else sample, axis=0)
    diffuse = np.minimum(under, flat)
    shadow = np.divide(diffuse, flat, out=np.ones_like(flat), where=flat > 0)
    headroom = 1 - diffuse
    highlight = np.divide(under - diffuse, headroom, out=np.zeros_like(flat), where=headroom > 0)
    return Stack(flat, shadow, highlight, ink)


def norm_blur(values, mask, sigma):
    w = mask.astype(np.float32)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return cv2.GaussianBlur(values*w, (0, 0), sigma) / np.maximum(den, 1e-5)


def light_direction(stack, groups):
    """One vote per well-supported diffuse surface, weighted by its contrast."""
    diffuse = cv2.cvtColor(stack.flat * stack.shadow, cv2.COLOR_RGB2GRAY)
    votes = []
    for g, sl in region_slices(groups, pad=2):
        m = (groups[sl] == g) & (stack.lines[sl][..., 3] < .1)
        if m.sum() < max(32, groups.size * .0004):
            continue
        y,x = np.where(m)
        samples = diffuse[sl][m]
        dark, lit = np.percentile(samples, [20, 80])
        if lit-dark < .025:
            continue
        xy = np.column_stack((x,y))
        span = np.maximum(np.ptp(xy, axis=0), 1)
        vote = (xy[samples >= lit].mean(0) - xy[samples <= dark].mean(0))/span
        votes.append(vote * np.sqrt(lit-dark))
    total = np.sum(votes, axis=0) if votes else np.zeros(2)
    norm = np.linalg.norm(total)
    confidence = norm / max(sum(np.linalg.norm(v) for v in votes), 1e-6)
    return total/max(norm, 1e-6), float(confidence)


def directed_bridge(density, allowed, radius):
    """Close short valleys along a fold tangent. Both endpoints must carry
    pigment; restrict new paint to the thin stroke corridor between them."""
    raised = density.copy()
    for angle in np.arange(0, np.pi, np.pi/8):
        dx, dy = np.rint(radius*np.array([np.cos(angle), np.sin(angle)])).astype(int)
        kernel = np.zeros((2*radius+1, 2*radius+1), np.uint8)
        cv2.line(kernel, (radius-dx,radius-dy), (radius+dx,radius+dy), 1)
        closed = cv2.morphologyEx(density, cv2.MORPH_CLOSE, kernel)
        np.maximum(raised, closed, out=raised)
    return density + allowed*(raised-density)


def fold_families(seed, field, width):
    """Join interrupted cores only if their principal axes agree and their
    nearest endpoints face each other across a short, narrow gap."""
    count, ids, stats, centres = cv2.connectedComponentsWithStats(seed)
    clusters = []
    for i in range(1,count):
        if stats[i,4] < max(4,width*.25):
            continue
        yy,xx = np.where(ids==i)
        pts = np.column_stack((xx,yy)).astype(np.float32)
        axis = cv2.PCACompute(pts, mean=None)[1][0]
        projected = pts[:, 0] * axis[0] + pts[:, 1] * axis[1]
        ends = pts[[projected.argmin(), projected.argmax()]]
        clusters.append((i,axis,ends,float(field[yy,xx].sum())))
    parent = list(range(len(clusters)))
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for j,(_,axis,ends,_) in enumerate(clusters):
        for i,(_,other,other_ends,_) in enumerate(clusters[:j]):
            if abs(axis@other) < .86:
                continue
            delta = (ends[:,None,:]-other_ends).reshape(-1,2)
            near = delta[np.argmin((delta**2).sum(1))]
            if np.linalg.norm(near) < width*.20 and abs(near@[-axis[1],axis[0]]) < width*.046:
                parent[root(i)] = root(j)
    families = {}
    for i,(label,_,_,score) in enumerate(clusters):
        families.setdefault(root(i),[]).append((label,score))
    ranked = sorted(families.values(), key=lambda f: sum(s for _,s in f),reverse=True)
    return [[lab for lab,_ in family] for family in ranked[:3]], ids


def fold_envelope(core, width):
    """Fit the two banks in the principal frame of a fold. Interpolate short
    gaps along the tangent and taper before either end. The envelope guides
    opacity; the observed paint supplies the edge profile inside it."""
    yy,xx = np.where(core)
    pts = np.column_stack((xx,yy)).astype(np.float32)
    centre, axes = cv2.PCACompute(pts,mean=None)
    # OpenCV evaluates this two-term projection in float32. Supply y then x
    # to retain the rounding order of the released x/y matrix product.
    yx = (pts - centre)[:, ::-1].reshape(-1, 1, 2)
    uv = cv2.transform(yx, axes[:, ::-1]).reshape(-1, 2)
    step = max(2.,width*.055)
    bins = np.floor((uv[:,0]-uv[:,0].min())/step).astype(int)
    u,lo,hi = [],[],[]
    for k in np.unique(bins):
        segment = uv[bins==k]
        u.append(float(segment[:,0].mean()))
        lo.append(np.percentile(segment[:,1],8))
        hi.append(np.percentile(segment[:,1],92))
    y,x = np.indices(core.shape,dtype=np.float32)
    x -= centre[0,0]
    y -= centre[0,1]
    tangent = x*axes[0,0]+y*axes[0,1]
    normal = x*axes[1,0]+y*axes[1,1]
    outside = np.maximum(np.interp(tangent,u,lo)-normal,normal-np.interp(tangent,u,hi))
    beyond = np.maximum(u[0]-tangent,tangent-u[-1])
    feather = max(2.,width*.048)
    distance = np.hypot(np.maximum(outside,0),np.maximum(beyond,0))
    return np.exp(-.5*(distance/feather)**2).astype(np.float32)


def paint_folds(old, mask, ink, width, orientation, confidence):
    density = 1 - cv2.cvtColor(old,cv2.COLOR_RGB2GRAY)
    radius = max(2,int(round(.065*width)))
    thick = cv2.distanceTransform(np.uint8(mask),cv2.DIST_L2,5)
    original = norm_blur(density,mask,1.0)
    depth = max(.035,float(np.percentile(density[mask],94)))
    # Find load-bearing cores away from the hem, so a dark outline cannot
    # turn several independent folds into one enormous connected component.
    core = np.uint8((original > .51*depth) & (thick > .07*width))
    erosion = max(1,int(round(.035*width)))
    core = cv2.erode(core,_ellipse(2*erosion+1))
    families,ids = fold_families(core, original*(1+.15*confidence*orientation), width)
    if not families:
        return old
    supports = [np.isin(ids,family) for family in families]
    masses = [original[k].sum() for k in supports]
    # Prefer two well-supported stacks; keep a third when the two largest
    # fail to explain most of the loaded fabric. This works in the fold frame.
    if len(supports)==3 and sum(masses[:2]) >= .70*sum(masses):
        supports = supports[:2]
    support = np.maximum.reduce([fold_envelope(k,width) for k in supports])
    corridor = cv2.dilate(ink,_ellipse(2*radius+1))
    # Keep the native hard side of a cast edge, smooth across the contour
    # only for coverage, and carry supported pigment through a narrow ink gap.
    antialiased = .70*density + .30*original
    joined = directed_bridge(antialiased,mask*corridor,radius)
    wide = norm_blur(original,mask,max(2.,width*.15))
    opacity = .09*wide + .62*support*joined
    dark = mask & (density>.45*depth)
    dye = np.median((1-old[dark])/density[dark,None],axis=0)
    # Local pigment survives where the same dye is supported over a small
    # area. A near-neutral crease stays neutral; a lilac fold stays lilac.
    local_mass = norm_blur(density,mask,max(2.,width*.055))
    local_dye = np.stack([norm_blur(1-old[...,c],mask,max(2.,width*.055))
                          for c in range(3)],axis=-1) / np.maximum(local_mass[...,None],.002)
    mix = .65*np.clip(local_mass/max(.04,depth*.35),0,1)
    dye = dye + mix[...,None]*(local_dye-dye)
    return np.clip(1-opacity[...,None]*dye,0,1).astype(np.float32)


def matte_fill_light(stack, new_shadow, groups, materials, protect, strength):
    """Rebase broad diffuse fill in absolute RGB. Near-white flats have tiny
    screen headroom: replaying their old coefficients on darker rebuilt paint
    otherwise magnifies subpixel pink/white compression fringes."""
    highlight = stack.highlight.copy()
    for g,sl in region_slices(groups,pad=3):
        if materials.get(g) != 'cloth_light':
            continue
        m = groups[sl]==g
        flat = stack.flat[sl]
        lift = (1-flat*stack.shadow[sl])*stack.highlight[sl]
        width = 2*np.percentile(cv2.distanceTransform(np.uint8(m),cv2.DIST_L2,5)[m],85)
        sigma = np.clip(width*.075,1.5,12)
        fill = np.stack([norm_blur(lift[...,c],m,sigma) for c in range(3)],axis=-1)
        amplitude = (strength*(1-protect[sl]))[...,None]
        target = np.clip(fill/np.maximum(1-flat*new_shadow[sl],1e-5),0,1)
        highlight[sl][m] += (amplitude*(target-stack.highlight[sl]))[m]
    return highlight


def rebuild_shadows(stack, groups, materials, protect, strength, light, confidence):
    shadow = stack.shadow.copy()
    for g, sl in region_slices(groups, pad=3):
        mat = materials.get(g)
        if mat in (None, 'eyes', 'gem', 'metal'):
            continue
        m = groups[sl] == g
        # Use thickness, not frame dimensions, to identify detail scales.
        dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
        width = 2 * np.percentile(dist[m], 85)
        if width < 5:
            continue
        r = np.clip(width * .16, 1.5, 24)
        old = stack.shadow[sl]
        density = 1 - cv2.cvtColor(old, cv2.COLOR_RGB2GRAY)
        fine = norm_blur(density, m, 1.3)
        broad = norm_blur(density, m, r)
        depth = max(.04, float(np.percentile(density[m], 94)))
        coherence = np.clip((broad/depth - .10)/.62, 0, 1)
        # Narrow oscillations lose amplitude; a broad, monotone ramp survives.
        # At a genuine cast edge retain the original transition profile.
        gx = cv2.Sobel(broad, cv2.CV_32F, 1, 0)/8
        gy = cv2.Sobel(broad, cv2.CV_32F, 0, 1)/8
        orientation = -(gx*light[0] + gy*light[1]) / np.maximum(np.hypot(gx,gy), .0001)
        agree = 1 - .22*confidence*np.maximum(-orientation,0)
        gain = {
            'cloth_light': .10 + .56*coherence**2,
            'cloth_dark': .24 + .48*coherence,
            'skin': .28 + .59*coherence,
            'leather_glossy': .18 + .42*coherence,
            'leather_strap': .72 + .24*coherence,
            'horn': .75 + .23*coherence,
            'hair': .56 + .36*coherence,
            'membrane': .86 + .13*coherence,
        }[mat] * agree
        new_density = gain*(.78*density+.22*fine)
        if mat == 'membrane':
            # Small oscillations only: neither quantise nor draw a new gradient.
            new_density = density - .48*(density-fine) * (1-coherence)
        ratio = np.clip(new_density / np.maximum(density,.001), 0, 1)
        rebuilt = 1 - (1-old)*ratio[...,None]
        if mat == 'cloth_light':
            rebuilt = paint_folds(old, m, stack.lines[sl][...,3], width, orientation, confidence)
        # SLIC boundaries without a visible pigment edge must never become ink.
        weight = (strength * (1-protect[sl]) * np.clip(dist/5, 0, 1))[...,None]
        shadow[sl][m] = (old + weight*(rebuilt-old))[m]
    return shadow


def reflection_on_seam(rgb, island, host, body):
    """A long glint between inked banks can resemble a separate outlined part.
    Check that both banks continue along the host beyond both tips."""
    y, x = np.nonzero(island)
    points = np.column_stack((x, y))
    centre = points.mean(axis=0)
    offsets = points - centre
    _, axes = np.linalg.eigh(offsets.T @ offsets)
    transverse, lengthwise = (offsets @ axes).T
    half_width = np.ptp(transverse) * .5
    if half_width < 1 or np.ptp(lengthwise) < 4.4 * half_width:
        return False
    bands = np.linspace(-half_width, half_width, 9)
    for sign, end in ((-1, lengthwise.min()), (1, lengthwise.max())):
        longitudinal = end + sign * np.array([.85, 1.40]) * half_width
        probes = (centre + longitudinal[:, None, None] * axes[:, 1] +
                  bands[None, :, None] * axes[:, 0])
        px, py = np.rint(probes).astype(int).transpose(2, 0, 1)
        if (px.min() < 0 or py.min() < 0 or px.max() >= host.shape[1] or
                py.max() >= host.shape[0] or not host[py, px].all()):
            return False
        tone = rgb[py, px].mean(axis=2)
        if np.any(tone.min(axis=1) >= .72 * body) or np.any(tone.max(axis=1) <= body + .08):
            return False
    return True


def attach_reflections(img, groups, materials, lines):
    """A broad, unoutlined light island enclosed by glossy paint shares its flat.
    Recognise it by its colour along the host-to-white mixing line. A punched
    ring has an interior contour; a jewel or buckle has a contrasting rim.
    """
    fixed = groups.copy()
    moved = np.zeros(groups.shape, bool)
    areas = np.bincount(groups.ravel())
    for g, sl in region_slices(groups, pad=8):
        if materials.get(g) in ('eyes', 'gem', 'metal', 'leather_glossy', 'leather_strap', 'horn'):
            continue
        mask = (groups[sl] == g).astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        for idx, contour in enumerate(contours):
            if hierarchy[0, idx, 3] >= 0 or hierarchy[0, idx, 2] >= 0:
                continue
            x,y,w,h = cv2.boundingRect(contour)
            if w*h > groups.size*.002 or min(w,h) < 3:
                continue
            tile = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(tile, [contour], 0, 1, -1)
            ring = (cv2.dilate(tile, _ellipse(9)) != 0) & (tile == 0)
            if not ring.any():
                continue
            ids = groups[sl][ring]
            host = np.bincount(ids).argmax()
            if materials.get(host) != 'leather_glossy' or np.mean(ids==host) < .78:
                continue
            part = tile.astype(bool)
            if part.sum() > .06*areas[host]:
                continue
            body = np.median(img[sl][ring & (groups[sl]==host)],0)
            # A black-hat next to a white glint fires even over ordinary brown
            # leather. A physical outline must also be dark relative to its host.
            rim_ink = lines[sl][ring] & (img[sl][ring].mean(1)<.72*body.mean())
            bright = np.median(img[sl][part],0)
            v = 1-body
            t = np.dot(bright-body,v) / max(np.dot(v,v), 1e-4)
            if t > .16 and np.linalg.norm(bright-(body+t*v)) < .18:
                if rim_ink.mean() > .32 and not reflection_on_seam(
                        img[sl], part, groups[sl] == host, body.mean()):
                    continue
                fixed[sl][part] = host
                moved[sl][part] = True
    return fixed, moved


def tangent_filter(opacity, value):
    """Subpixel averaging along a stroke, using the local structure tensor."""
    dx = cv2.Sobel(value, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(value, cv2.CV_32F, 0, 1)
    blur = lambda v: cv2.GaussianBlur(v,(0,0),1.5)
    angle = .5*np.arctan2(2*blur(dx*dy), blur(dx*dx-dy*dy))
    tx, ty = -np.sin(angle), np.cos(angle)
    y,x = np.indices(value.shape,dtype=np.float32)
    smooth = .4*opacity
    for t,w in ((-.9,.22),(.9,.22),(-1.8,.08),(1.8,.08)):
        smooth += w*cv2.remap(opacity, x+t*tx,y+t*ty,cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return smooth


def cloth_ink(stack, new_shadow, groups, materials, tiers, protect, strength):
    """Rebase cloth crease ink on the repainted surface. Clear bright fringe
    ink and transfer only continuous, two-sided dark pigment to the new flat."""
    from masks import TIER_INTRA
    out = stack.lines.copy()
    original = stack.composite()
    tone = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    for g,sl in region_slices(groups,pad=5):
        if materials.get(g) not in ('cloth_light','cloth_dark'):
            continue
        m = groups[sl] == g
        crease = m & (tiers[sl] == TIER_INTRA)
        if not crease.any():
            continue
        if materials.get(g)=='cloth_dark':
            # Apply the same local affine pigment change to ink and underpaint.
            # The original antialiasing then crosses the crease continuously.
            a = 1 - cv2.cvtColor(stack.shadow[sl],cv2.COLOR_RGB2GRAY)
            b = 1 - cv2.cvtColor(new_shadow[sl],cv2.COLOR_RGB2GRAY)
            gain = np.clip(norm_blur(b,m,3)/np.maximum(norm_blur(a,m,3),.002),0,1)
            zone = cv2.dilate(crease.astype(np.uint8),_ellipse(5)).astype(np.float32)
            zone *= strength
            zone *= ((tiers[sl]==0) | (tiers[sl]==TIER_INTRA))*m*(1-protect[sl])
            ink = stack.lines[sl][...,:3]
            out[sl][...,:3] += ((1-gain)*zone)[...,None]*np.maximum(stack.flat[sl]-ink,0)
            continue
        s = tone[sl]
        blackhat = cv2.morphologyEx(s,cv2.MORPH_BLACKHAT,_ellipse(7))
        ref = stack.flat[sl]
        body = cv2.cvtColor(ref,cv2.COLOR_RGB2GRAY)
        # Smooth confidence avoids the jagged boundary of a binary ink mask.
        # Transfer measured local ink chromaticity, keeping stroke strength
        # independent of hue. A painted shadow edge needs two brighter banks
        # before it can add a thin stroke to the rebuilt surface.
        source_rgb = original[sl]
        local_ink = source_rgb / np.maximum(s[...,None],.08)
        ink_color = np.clip(.43*body[...,None]*local_ink,0,1)
        blackhat = np.minimum(blackhat,paired_banks(s,3))
        opacity = np.clip(blackhat/np.maximum(.57*body,.04),0,1)
        persistent = cv2.GaussianBlur(opacity,(0,0),1.1)
        opacity *= np.clip(persistent/.16,0,1)
        opacity = tangent_filter(opacity,s)
        active = cv2.dilate(crease.astype(np.uint8),_ellipse(7)).astype(np.float32)
        active *= (tiers[sl]==0) | (tiers[sl]==TIER_INTRA)
        distance = cv2.distanceTransform(m.astype(np.uint8),cv2.DIST_L2,5)
        interior = np.clip((distance-1.5)/3.0,0,1)
        weight = strength*(1-protect[sl])*active*interior
        old = stack.lines[sl]
        alpha = (1-weight)*old[...,3]+weight*opacity
        premul = ((1-weight)*old[...,3])[...,None]*old[...,:3] + (weight*opacity)[...,None]*ink_color
        out[sl][m] = np.dstack((premul/np.maximum(alpha[...,None],1e-6),alpha))[m]
    return out
