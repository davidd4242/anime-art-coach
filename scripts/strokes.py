"""Subpixel stroke fitting and coverage for pressure-varied illustrator ink."""
import cv2
import numpy as np
from skimage.morphology import skeletonize

from masks import _ellipse, TIER_OUTLINE, TIER_INTER, TIER_INTRA
from pigment import paired_banks


def paths_on_skeleton(skeleton):
    """Split an eight-connected skeleton at genuine junctions. A diagonal
    shortcut around an already connected right angle is not a third branch."""
    y, x = np.nonzero(skeleton)
    if not len(x):
        return
    labels = np.full(np.array(skeleton.shape)+2,-1,np.int32)
    labels[y+1,x+1] = np.arange(len(x))
    adjacent = [[] for _ in x]
    for dy, dx in ((0,1),(1,0),(1,1),(1,-1)):
        other = labels[y+1+dy,x+1+dx]
        valid = other >= 0
        if dx and dy:
            valid &= (labels[y+1,x+1+dx]<0) & (labels[y+1+dy,x+1]<0)
        for a,b in zip(np.flatnonzero(valid).tolist(),other[valid].tolist()):
            adjacent[a].append(b)
            adjacent[b].append(a)
    degree = np.array([len(n) for n in adjacent])
    seen = set()
    coords = np.column_stack((x,y)).astype(np.float32)
    # Open paths first. The remaining edges form closed components.
    for a in np.r_[np.flatnonzero(degree!=2),np.flatnonzero(degree==2)].tolist():
        for b in adjacent[a]:
            if (a,b) in seen:
                continue
            chain = [a]
            prev,cur = a,b
            while True:
                seen.add((prev,cur))
                seen.add((cur,prev))
                chain.append(cur)
                if cur==a or degree[cur]!=2:
                    break
                left,right = adjacent[cur]
                prev,cur = cur,right if left==prev else left
            yield coords[chain], degree[chain[0]], degree[chain[-1]]


def arc_smooth(values, sigma, closed):
    radius = max(1,int(np.ceil(3*sigma)))
    d = np.arange(-radius,radius+1,dtype=np.float32)
    kernel = np.exp(-.5*(d/sigma)**2)
    mode = 'wrap' if closed else 'edge'
    return np.convolve(np.pad(values,radius,mode=mode),kernel/kernel.sum(),mode='valid')


def stroke_shapes(evidence, core, pressure, antialias=4):
    """Fit a continuous strip to each ridge. Interpolate its two banks on an
    oversampled grid. Only free ends taper; junction anchors stay connected."""
    shape = core.shape
    bitmap = np.zeros((antialias*shape[0],antialias*shape[1]),np.uint8)
    if not core.any():
        return np.zeros(shape,np.float32)
    radius = cv2.distanceTransform(core.astype(np.uint8),cv2.DIST_L2,5)
    # Subpixel centroid corrects the one-pixel steps in a raster skeleton.
    sample = np.maximum(evidence,.001)*core
    den = cv2.GaussianBlur(sample,(0,0),.9)
    gx = cv2.Sobel(den,cv2.CV_32F,1,0)/8
    gy = cv2.Sobel(den,cv2.CV_32F,0,1)/8
    for points,first,last in paths_on_skeleton(skeletonize(np.ascontiguousarray(core))):
        if len(points)<2:
            continue
        ix,iy = points.astype(int).T
        rad = np.maximum(.36,radius[iy,ix]-.43)*pressure[iy,ix]
        offset = .81*np.column_stack((gx[iy,ix],gy[iy,ix]))/np.maximum(den[iy,ix,None],.005)
        points = points+np.clip(offset,-.45,.45)
        closed = np.array_equal(points[0],points[-1])
        arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        length = arc[-1]
        if length<1e-4:
            continue
        u = np.linspace(0,length,max(3,int(np.ceil(length*1.6))+1),endpoint=not closed)
        xy = np.array([np.interp(u,arc,c) for c in points.T])
        span = np.clip(.095*len(u),1.35,6.0)
        fit = np.vstack([arc_smooth(c,span,closed) for c in xy])
        shift = fit-xy
        track = xy+shift/np.maximum(1,np.linalg.norm(shift,axis=0)/.85)
        typical = np.median(rad)
        widths = arc_smooth(np.interp(u,arc,np.clip(rad,.55*typical,1.55*typical)),span,closed)
        # Lock multi-branch anchors. Free ends decrease linearly in radius,
        # hence continuously in pixel coverage even below one pixel wide.
        if not closed:
            for pos,deg in ((0,first),(-1,last)):
                distance = u if pos==0 else length-u
                lock = np.exp(-distance/1.25)
                track += (xy[:,pos]-track[:,pos])[:,None]*lock
                if deg==1:
                    tip = np.clip(distance/min(max(2.5,3*widths[pos]),.46*length),0,1)
                    widths *= .16+.84*tip
        tang = np.roll(track,-1,axis=1)-np.roll(track,1,axis=1) if closed else np.gradient(track,axis=1)
        tang /= np.maximum(np.linalg.norm(tang,axis=0),1e-6)
        norm = np.array([-tang[1],tang[0]])*widths
        # The first and last cross-sections coincide for a closed strip.
        if closed:
            track = np.c_[track,track[:,0]]
            norm = np.c_[norm,norm[:,0]]
        poly = np.vstack(((track+norm).T,(track-norm).T[::-1]))
        # Pixel centres in the native grid map to the centre of their
        # antialias cell. Use fixed-point polygon coordinates for fractional tips.
        poly = np.rint(((poly+.5)*antialias-.5)*16).astype(np.int32)
        cv2.fillPoly(bitmap,[poly],255,shift=4)
        if not closed:
            for pos,deg in ((0,first),(-1,last)):
                if deg>2:
                    c = np.rint(((track[:,pos]+.5)*antialias-.5)*16).astype(int)
                    cv2.circle(bitmap,tuple(c),round(max(.2,widths[pos])*antialias*16),255,-1,shift=4)
    return cv2.resize(bitmap,shape[::-1],interpolation=cv2.INTER_AREA).astype(np.float32)/255


def line_colour(rgb, weights, sigma):
    den = cv2.GaussianBlur(weights,(0,0),sigma)
    return cv2.GaussianBlur(rgb*weights[...,None],(0,0),sigma)/np.maximum(den[...,None],1e-6)


def finish_lines(stack, src, tiers, groups, materials, face, silhouette, protect, strength, light):
    """Fit the existing outer and structural ink on its lifted underpainting.
    Use the original material colour locally and preserve face and small fittings."""
    held = np.isin(groups,[g for g,m in materials.items() if m in ('eyes','metal','gem')]) | face
    leather = np.isin(groups,[g for g,m in materials.items() if m=='leather_glossy'])
    outline = tiers==TIER_OUTLINE
    core_allowed = ~held & (outline | ((tiers==TIER_INTER)|(tiers==TIER_INTRA)) & ~leather)
    old = stack.lines
    under = stack.flat*stack.shadow
    under += (1-under)*stack.highlight
    delta = cv2.cvtColor(under-old[...,:3],cv2.COLOR_RGB2GRAY)
    core = core_allowed & (old[...,3]>.32) & (delta>.035)
    if not core.any():
        return old
    y,x = np.where(core)
    sl = np.s_[max(0,y.min()-4):y.max()+5,max(0,x.min()-4):x.max()+5]
    outer = outline[sl]
    signed = cv2.GaussianBlur(cv2.distanceTransform(silhouette[sl].astype(np.uint8),cv2.DIST_L2,5),(0,0),2)
    dx = cv2.Sobel(signed,cv2.CV_32F,1,0)
    dy = cv2.Sobel(signed,cv2.CV_32F,0,1)
    facing = -(dx*light[0]+dy*light[1])/np.maximum(np.hypot(dx,dy),1e-5)
    pressure = np.where(outer,1.07-.16*facing,np.where(tiers[sl]==TIER_INTER,.85,.68)).astype(np.float32)
    weight = np.clip(delta[sl],0,1)*old[sl][...,3]*core[sl]
    coverage = stroke_shapes(weight,core[sl],pressure)
    gate = cv2.dilate(core[sl].astype(np.uint8),_ellipse(3)).astype(np.float32)
    # Where lifting did not establish a reliable new surface, mix gently.
    # Avoid laying a filled colour across a silhouette or a metal's edge.
    gate *= (~held[sl])*(1-protect[sl])*strength
    gate *= np.where(cv2.dilate(outer.astype(np.uint8),_ellipse(3))>0,.43,.34)
    gate *= (~leather[sl]) | (cv2.dilate(outer.astype(np.uint8),_ellipse(3))>0)
    # Do not wash a fully inked silhouette centre out with its lighter
    # antialias shoulders. Its original dark pigment anchors the fitted strip.
    outer_band = cv2.dilate(outer.astype(np.uint8),_ellipse(3))>0
    darkness = np.maximum(delta[sl],0)/np.maximum(cv2.cvtColor(under[sl],cv2.COLOR_RGB2GRAY),.05)
    anchor = np.clip((darkness-.17)/.54,0,1)*old[sl][...,3]
    gate *= 1-.83*anchor*outer_band
    colour = line_colour(src[sl],weight,1.3)
    previous = old[sl]
    alpha = previous[...,3]*(1-gate)+coverage*gate
    premul = previous[...,:3]*(previous[...,3]*(1-gate))[...,None] + colour*(coverage*gate)[...,None]
    new = old.copy()
    active = gate>0
    new[sl][active] = np.dstack((premul/np.maximum(alpha[...,None],1e-6),alpha))[active]
    return new


def lace_contacts(hardware):
    """Two repeated rails of eyelets bound woven straps. Intersect adjacent
    pairs in their own coordinate frame; spend pressure only on observed
    recessed pigment near the crossing, leaving the upper strap continuous."""
    n,_,stats,centres = cv2.connectedComponentsWithStats(np.uint8(hardware>.52))
    out = np.zeros_like(hardware)
    panel = np.zeros_like(out)
    if n<5:
        return out,panel
    area = stats[1:,cv2.CC_STAT_AREA]
    centres = centres[1:][(area>=.45*np.median(area)) & (area<=1.9*np.median(area))]
    if len(centres)<4:
        return out,panel
    delta = centres[:,None]-centres[None,:]
    distances = np.linalg.norm(delta,axis=-1)
    np.fill_diagonal(distances,np.inf)
    near = distances.argmin(1)
    pitch = np.median(distances.min(1))
    if pitch<3:
        return out,panel
    directions = centres-centres[near]
    _,axis = np.linalg.eigh(directions.T@directions)
    along = axis[:,-1]
    across = np.array([-along[1],along[0]])
    s = centres@along
    order = np.argsort(s)
    rows = np.split(order,np.flatnonzero(np.diff(s[order])>.43*pitch)+1)
    pairs = [centres[r[np.argsort(centres[r]@across)]] for r in rows if len(r)==2]
    y,x = np.indices(hardware.shape,dtype=np.float32)
    for first,second in zip(pairs,pairs[1:]):
        separation=(second.mean(0)-first.mean(0))@along
        if not .55*pitch<separation<1.6*pitch:
            continue
        d1,d2 = second[1]-first[0],second[0]-first[1]
        matrix = np.column_stack((d1,-d2))
        if abs(np.linalg.det(matrix))<1:
            continue
        t,_ = np.linalg.solve(matrix,first[1]-first[0])
        centre = first[0]+t*d1
        across_pos=(x-centre[0])*across[0]+(y-centre[1])*across[1]
        along_pos=(x-centre[0])*along[0]+(y-centre[1])*along[1]
        falloff=(across_pos/(pitch*.43))**2+(along_pos/(pitch*.27))**2
        out=np.maximum(out,np.exp(-.5*falloff).astype(np.float32))
        quadrilateral = np.rint(np.vstack((first,second[::-1]))).astype(np.int32)
        cv2.fillConvexPoly(panel,quadrilateral,1.)
    panel = cv2.GaussianBlur(panel,(0,0),max(1.,.13*pitch))
    return out,panel


def leather_ink(src, original_under, tone, body, painted, old_under, accent, hardware, part, dist, width):
    """Erase unsupported narrow valleys from the matte plate and reconstruct
    continuous coloured seams from the original ridges, with tapered ends."""
    clean = cv2.GaussianBlur(tone,(0,0),.65)
    radius = float(np.clip(width * .038, 2, 5))
    smaller, larger = int(np.floor(radius)), int(np.ceil(radius))
    depth = paired_banks(clean, smaller)
    if larger > smaller:
        depth += (radius - smaller) * (paired_banks(clean, larger) - depth)
    # Discount the outline of a white reflection without discarding a genuine
    # dark seam which passes through the same area.
    body_l = cv2.cvtColor(body,cv2.COLOR_RGB2GRAY)
    pigment = np.clip((body_l+.075-clean)/.095,0,1)
    evidence = cv2.GaussianBlur(np.minimum(depth,.24)*pigment,(0,0),.52)
    valid = part & (dist>1.1) & (hardware<.52)
    core = (evidence>.029) & valid
    # Bridge a subpixel dropout only when both a neighbouring core and
    # observed weak pigment support it. A genuine open tip still tapers.
    connected = cv2.morphologyEx(core.astype(np.uint8),cv2.MORPH_CLOSE,_ellipse(5))
    core |= (connected>0) & (evidence>.016) & valid
    n,labels,stats,_ = cv2.connectedComponentsWithStats(np.uint8(core))
    keep = stats[:,4]>=max(4.,.075*width)
    keep[0]=False
    core = keep[labels]
    if not core.any():
        return old_under,painted,np.zeros_like(tone)
    contacts,panel = lace_contacts(hardware)
    # The original dark bank tells which strand is underneath. A lit face of
    # the overpassing strand receives no contact ink even at the geometric X.
    recess = np.clip((body_l-clean)/.11,0,1)
    pressure = .82+.23*contacts*recess
    coverage = stroke_shapes(evidence,core,pressure)
    reliable = np.clip(evidence/.15,0,1)*core
    ink = line_colour(src,reliable,3.3)
    ink_l = cv2.cvtColor(ink,cv2.COLOR_RGB2GRAY)
    hold = np.minimum(ink_l,body_l-.045-.036*contacts*recess)
    ink *= np.clip(hold/np.maximum(ink_l,.025),.58,1)[...,None]
    # A fitted line still has a soft foot where its pigment meets the paint.
    foot = cv2.GaussianBlur(coverage,(0,0),.77)
    alpha = np.clip(.85*coverage+.15*foot,0,1)
    # Remove the pigment of rejected satellite marks from the underlying
    # plate too. Restoring only the fitted ridge leaves a parallel ghost.
    dark = np.minimum(clean-body_l,0)
    cast = cv2.GaussianBlur(dark,(0,0),2.4)
    matte = body*(1+.34*(cast/np.maximum(body_l,.06))[...,None])
    # Woven faces have their own diffuse midtone. Transfer the two broad banks
    # of a ribbon at a crossover, retaining continuity of its upper strand.
    # The deeper bank alone gains extra contact shade.
    def ribbon_colour(rgb):
        diffuse = cv2.GaussianBlur(rgb,(0,0),.48)
        difference = diffuse-body
        return body+.78*np.minimum(difference,0)+.067*np.tanh(np.maximum(difference,0)/.09)
    contact_paint = np.clip((body_l-clean-.025)/.13,0,1)*contacts
    woven = ribbon_colour(src)*(1-.22*contact_paint)[...,None]
    surface = matte+panel[...,None]*(ribbon_colour(original_under)-matte)
    matte += panel[...,None]*(woven-matte)
    # Blend the fitted line most strongly on plain leather. Ribbon contours
    # already carry some of their ink in the transferred banks.
    alpha *= 1-.74*panel
    rebuilt = matte+(ink-matte)*alpha[...,None]
    rebuilt += .70*accent[...,None]*np.maximum(src-rebuilt,0)
    gate = np.clip((dist-1)/2,0,1)*(1-hardware)*part
    surface += .70*accent[...,None]*np.maximum(original_under-surface,0)
    visible = painted+gate[...,None]*(rebuilt-painted)
    surface = old_under+gate[...,None]*(surface-old_under)
    return np.clip(surface,0,1),np.clip(visible,0,1),alpha*gate
