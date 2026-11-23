"""Separate a material's specular accent from broad diffuse light and hardware."""
import cv2
import numpy as np

from masks import _ellipse, region_slices
from layers import norm_blur
from strokes import leather_ink
from colorfix import to_lab
from pigment import colour_support, paired_banks, restore_chroma


def hardware_coverage(v, part, distance, width):
    """Small curved metallic rims border a dark aperture. Identify the rim on
    its own geometry before ranking any of the surrounding leather reflections."""
    body = np.median(v[part])
    bright = np.uint8(part & (v > body + .16))
    n, ids, stats, _ = cv2.connectedComponentsWithStats(bright)
    mask = np.zeros(part.shape,np.uint8)
    for i in range(1,n):
        x,y,w,h,area = stats[i]
        diameter = max(w,h)
        if (min(w,h)<3 or diameter>max(4.,width*.19) or
                not .13 < area/(w*h) < .65 or min(w,h)/diameter<.32):
            continue
        cut = np.s_[y:y+h,x:x+w]
        arc = ids[cut]==i
        tile = v[cut]
        if (distance[cut][arc].mean()<diameter*.65 or
                tile[arc].max() < body+.33 or np.percentile(tile,12) > body*.93):
            continue
        # An eyelet turns around its aperture. A nearly straight bright
        # stroke can span a diagonal bounding box just as wide as a ring.
        row, col = np.nonzero(arc)
        transverse, longitudinal = np.linalg.eigvalsh(np.cov(col, row))
        if longitudinal > 16 * transverse:
            continue
        mask[cut][arc] = 1
    core = cv2.dilate(mask,_ellipse(5))
    return np.maximum(core,cv2.GaussianBlur(core.astype(np.float32),(0,0),1.0))


def primary_accent(signal, eligible, width, light, hair):
    """Rank coherent positive reflections against the local diffuse estimate.
    For hair use the shared envelope across intervening strand ink."""
    if not eligible.any():
        return np.zeros(signal.shape,np.float32)
    smooth = cv2.GaussianBlur(signal,(0,0),.90)
    level = max(.025,.45*float(np.percentile(smooth[eligible],98)))
    binary = np.uint8(eligible & (smooth > level))
    if hair:
        radius = max(1,int(round(.11*width)))
        binary = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,_ellipse(2*radius+1))
    n,cc,stats,centres = cv2.connectedComponentsWithStats(binary)
    if n<2:
        return np.zeros_like(signal)
    mass = np.bincount(cc.ravel(), weights=smooth.ravel(), minlength=n)
    ys,xs = np.where(eligible)
    pos = (centres-[xs.mean(),ys.mean()]) / np.maximum([np.ptp(xs),np.ptp(ys)],1)
    score = mass*(1+.30*np.clip(pos@light,-1,1))
    score[(stats[:,4]<max(3,eligible.sum()*.0003))] = 0
    score[0]=0
    if score.max()<=0:
        return np.zeros_like(signal)
    win = np.uint8(cc==score.argmax())
    # The core comes from the source; soft coverage trims its ragged extremities.
    outer = cv2.dilate(win,_ellipse(5))
    return cv2.GaussianBlur(outer.astype(np.float32),(0,0),1.1)


def apply_paint(flat, desired, visible, previous_alpha, preferred_diffuse):
    """Encode two colours, the painted surface and the observed ink-over-paint
    colour, into feasible Multiply / Screen / straight RGBA values."""
    diffuse = np.minimum(np.minimum(desired, flat), preferred_diffuse)
    shadow = np.divide(diffuse,flat,out=np.ones_like(flat),where=flat>1e-6)
    screen = np.divide(desired-diffuse,1-diffuse,out=np.zeros_like(flat),where=diffuse<1-1e-6)
    delta = visible-desired
    bound = np.where(delta<0,-delta/np.maximum(desired,1e-6),
                     delta/np.maximum(1-desired,1e-6)).max(2)
    alpha = np.maximum(previous_alpha,bound).clip(0,1)
    rgb = np.clip(desired+delta/np.maximum(alpha[...,None],1e-6),0,1)
    rgb[alpha==0] = 0
    return shadow,screen,np.dstack((rgb,alpha))


def leather_paint(src, raw_under, flat, part, dist, width, hardware, light):
    """Rank light on the observed leather, then map the painted surface and
    its antialiased strokes through one continuous curve."""
    tone = cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)
    body = np.percentile(tone[part], [12, 76]).astype(np.float32)
    sigma = np.clip(.21*width,3.,22.)
    diffuse = norm_blur(np.clip(tone,*body), part & (hardware<.25),sigma)
    signal = np.maximum(tone-diffuse,0)
    eligible = part & (dist>max(2.,.073*width)) & (hardware<.08)
    accent = np.zeros_like(tone)
    if eligible.any():
        smoothed = cv2.GaussianBlur(signal,(0,0),.85)
        cutoff = max(.09,.63*np.percentile(smoothed[eligible],99))
        seeds = np.uint8(eligible & (smoothed>cutoff))
        n,ids,stats,centres = cv2.connectedComponentsWithStats(seeds)
        thickness = cv2.distanceTransform(seeds,cv2.DIST_L2,5)
        mass = np.bincount(ids.ravel(),weights=(thickness*smoothed**2).ravel(),minlength=n)
        yy,xx = np.where(part)
        pos = (centres-[xx.mean(),yy.mean()])/np.maximum([np.ptp(xx),np.ptp(yy)],1)
        mass *= 1+.2*np.clip(pos@light,-1,1)
        mass[stats[:,4]<max(4.,.0003*part.sum())]=0
        mass[0]=0
        if mass.max()>0:
            core = np.uint8(ids==mass.argmax())
            accent = cv2.GaussianBlur(cv2.dilate(core,_ellipse(3)).astype(np.float32),(0,0),.95)
    base_l = cv2.cvtColor(flat,cv2.COLOR_RGB2GRAY)
    # Cast light over the curved last before applying a single specular lobe.
    base = flat+(diffuse-base_l)[...,None]*.78
    # Recover the broad turning planes of the tongue and rubber wall. Exclude
    # hardware and the specular tail before estimating local body colour.
    mids = part & (hardware<.20)
    if mids.any():
        lo,hi = np.percentile(tone[mids],[5,87]).astype(np.float32)
        capped = np.clip(tone,lo,hi)
        diffuse_rgb = src*(capped/np.maximum(tone,.025))[...,None]
        broad_rgb = np.stack([norm_blur(diffuse_rgb[...,c],mids,max(4,width*.12))
                              for c in range(3)],axis=-1)
        base += .73*(broad_rgb-base)
    source_valley = paired_banks(tone,3)
    stroke_cover = cv2.GaussianBlur(np.clip(source_valley/.12,0,1),(0,0),.95)
    def tone_map(rgb):
        # Keep dark pigment using two banks. Diffuse volume comes from a
        # scale wider than a lace. Limit the remaining close-range ripple
        # symmetrically, or a dark trough can create a second apparent glint.
        lum = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        ripple = .014*np.tanh((rgb-base)/.075)
        matte = base+ripple
        # Coverage is scalar. Mix towards the actual darker RGB rather than
        # subtracting chromatic pigment channel by channel from a brown flat,
        # which would invert burgundy lace ink to a grey-green fringe.
        darker = np.clip((cv2.cvtColor(matte,cv2.COLOR_RGB2GRAY)-lum)/.055,0,1)
        coverage = darker*(.16+.80*stroke_cover)
        matte += coverage[...,None]*(rgb-matte)
        return np.clip(matte+.70*accent[...,None]*np.maximum(rgb-matte,0),0,1)
    under = tone_map(raw_under)
    return leather_ink(src,raw_under,tone,base,tone_map(src),under,accent,hardware,part,dist,width)


def rebuild_highlights(raw, painted, src, groups, materials, protect, strength, light):
    """Return Screen and ink layers. Subtract specular light in absolute RGB,
    including bright spill trapped in the antialiased fringe of extracted ink."""
    shadows = painted.shadow.copy()
    out = painted.highlight.copy()
    ink = painted.lines.copy()
    v = cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)
    for g,sl in region_slices(groups,pad=4):
        mat = materials.get(g)
        if mat not in ('leather_glossy','hair','metal','gem','cloth_dark'):
            continue
        count, pieces, stats, _ = cv2.connectedComponentsWithStats(np.uint8(groups[sl]==g))
        for i in range(1,count):
            if stats[i,4]<20:
                continue
            part = pieces==i
            dist = cv2.distanceTransform(np.uint8(part),cv2.DIST_L2,5)
            width = 2*float(np.percentile(dist[part],85))
            if width<3:
                continue
            headroom = 1-raw.flat[sl]*raw.shadow[sl]
            lift = headroom*raw.highlight[sl]
            signal = cv2.cvtColor(lift, cv2.COLOR_RGB2GRAY)
            hardware = hardware_coverage(v[sl],part,dist,width) if mat=='leather_glossy' else np.zeros_like(dist)
            if mat=='leather_glossy':
                flat = raw.flat[sl]
                original_under = flat*raw.shadow[sl]+lift
                under,visible,fitted_alpha = leather_paint(src[sl],original_under,flat,part,dist,width,hardware,light)
                pigment = colour_support(src[sl],part,np.clip(width*.12,2,12))
                visible = restore_chroma(visible,to_lab(src[sl]),.72*pigment)
                original_lab = to_lab(original_under)
                under = restore_chroma(under,original_lab,.72*pigment)
                # An etched aperture is a separate object with its own metal
                # rim. Mix both colour fields to preserve coverage at its edge.
                rim = hardware[...,None]
                under += rim*(original_under-under)
                visible += rim*(src[sl]-visible)
                edge = np.clip((dist-1)/2,0,1)
                weight = (strength*(1-protect[sl])*edge)[...,None]
                prev_d = flat*painted.shadow[sl]
                prev_u = prev_d+(1-prev_d)*painted.highlight[sl]
                prev_a = painted.lines[sl][...,3:]
                prev_v = prev_u*(1-prev_a)+painted.lines[sl][...,:3]*prev_a
                under = prev_u + weight*(under-prev_u)
                visible = prev_v + weight*(visible-prev_v)
                stroke_alpha = fitted_alpha+hardware*(raw.lines[sl][...,3]-fitted_alpha)
                stroke_alpha = prev_a[...,0]+weight[...,0]*(stroke_alpha-prev_a[...,0])
                sh,hl,rgba = apply_paint(flat,under,visible,stroke_alpha,np.minimum(prev_d,under))
                edit = part & (weight[...,0]>0)
                shadows[sl][edit],out[sl][edit],ink[sl][edit] = sh[edit],hl[edit],rgba[edit]
                continue
            eligible = part & (dist>1.5) & (hardware<.1)
            if mat=='cloth_dark':
                accent = np.zeros_like(signal)
            else:
                accent = primary_accent(signal,eligible,width,light,mat=='hair')
            sigma = np.clip(width*.12,2.,12.)
            # Cap the upper tail before measuring broad illumination: a glint
            # removed from Screen must not leave a glowing blur in the diffuse.
            limit = np.percentile(lift[part],72,axis=0)
            broad = np.stack([norm_blur(np.minimum(lift[...,c],limit[c]),part,sigma)
                              for c in range(3)],axis=-1)
            fraction = .55 if mat in ('hair','metal','gem') else .28
            broad *= fraction
            target = broad + .70*accent[...,None]*np.maximum(lift-broad,0)
            new_base = painted.flat[sl]*painted.shadow[sl]
            screen = np.clip(target/np.maximum(1-new_base,1e-6),0,1)
            # Keep the subpixel boundary against the background, including
            # the original colour of tiny independent metal parts.
            edge = np.clip((dist-.6)/2,0,1)
            weight = strength*(1-protect[sl])*(1-hardware)*edge
            out[sl][part] += (weight[...,None]*(screen-raw.highlight[sl]))[part]
            if mat in ('hair','cloth_dark','gem','metal'):
                # Keep coherent coloured greys and small dyed bands under the
                # newly simplified light. Do not add pigment to a neutral area.
                support = colour_support(src[sl],part,sigma)
                w = .8 * support * weight
                if np.any(w>.015):
                    under = new_base+(1-new_base)*out[sl]
                    corrected = restore_chroma(under,to_lab(raw.flat[sl]*raw.shadow[sl]+lift),w)
                    spec = np.clip((corrected-new_base)/np.maximum(1-new_base,1e-6),0,1)
                    out[sl][part] = spec[part]
    return shadows,out,ink
