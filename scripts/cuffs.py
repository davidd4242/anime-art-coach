"""Simplify transverse folds on a matte cloth band that wraps an exposed limb."""

import cv2
import numpy as np


def limb_bands(groups, materials):
    """A wrap borders skin at both ends of its axis and open space at its sides."""
    cloth = np.isin(groups, [g for g, material in materials.items() if material == 'cloth_dark'])
    skin = np.isin(groups, [g for g, material in materials.items() if material == 'skin'])
    count, components, bounds, _ = cv2.connectedComponentsWithStats(cloth.astype(np.uint8))
    for index in range(1, count):
        x, y, width, height, area = [int(value) for value in bounds[index]]
        if area < 360 or min(width, height) < 17:
            continue
        roi = np.s_[max(y - 4, 0):y + height + 4, max(x - 4, 0):x + width + 4]
        piece = components[roi] == index
        edge = cv2.dilate(piece.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~piece
        skin_edge = edge & skin[roi]
        if (edge & (groups[roi] == 0)).sum() < .33 * edge.sum() or skin_edge.sum() < .16 * edge.sum():
            continue
        points = np.column_stack(np.nonzero(piece)[::-1]).astype(np.float32)
        center = points.mean(0)
        values, vectors = np.linalg.eigh(np.cov(points.T))
        ratio = np.sqrt(values[1] / max(values[0], 1e-4))
        if not 1.17 <= ratio <= 2.75:
            continue
        axis = vectors[:, 1].astype(np.float32)
        if axis[1] < 0:
            axis = -axis
        across = np.array((axis[1], -axis[0]))
        basis = np.column_stack((across, axis))
        extents = np.einsum("ni,ij->nj", points - center, basis)
        width_span, length_span = np.ptp(extents, axis=0)
        neighbor_points = np.column_stack(np.nonzero(skin_edge)[::-1]).astype(np.float32)
        neighbors = np.einsum("ni,ij->nj", neighbor_points - center, basis)
        lateral, along = (neighbors / (width_span, length_span)).T
        ends = np.array([((s * along > .31) & (np.abs(lateral) < .34)).sum() for s in (-1, 1)])
        if min(ends) < max(5, .075 * edge.sum()) or ends.sum() < .73 * len(neighbors):
            continue
        lower = np.floor(extents.min(0))
        upper = np.ceil(extents.max(0))
        matrix = np.column_stack((basis, center + (basis * lower).sum(1)))
        size = tuple(int(v) for v in upper - lower + 1)
        yield roi, piece, matrix, size


def warp_into(values, matrix, size, nearest=False):
    mode = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(values, matrix, size, flags=mode | cv2.WARP_INVERSE_MAP)


def wrap_separators(source, mask):
    """Trace two source-backed lateral dark banks separating the wrapped faces."""
    height, width = mask.shape
    interior = cv2.erode(mask, np.ones((3, 3), np.uint8)).astype(np.float32)
    core = interior > 0
    if not core.any():
        return None
    luma = cv2.cvtColor(source, cv2.COLOR_RGB2Lab)[..., 0]
    if np.percentile(luma[core], 60) > 57:
        return None
    den = cv2.GaussianBlur(interior, (0, 0), 1.2, sigmaY=1.)
    fine = cv2.GaussianBlur(luma * interior, (0, 0), 1.2, sigmaY=1.) / np.maximum(den, .01)
    step = max(3, round(width * .085))
    index = np.arange(height)
    above = fine[np.maximum(index - step, 0)]
    below = fine[np.minimum(index + step, height - 1)]
    dark = np.maximum(0, np.median(luma[core]) - fine)
    bank = (np.maximum(0, np.maximum(above, below) - fine) +
            1.25 * np.maximum(0, np.minimum(above, below) - fine))
    bank *= np.clip((dark + 1) / 4, .1, 1.3) * interior
    local = cv2.dilate(bank, np.ones((3, 1), np.uint8))
    columns = np.arange(round(width * .10), round(width * .90))
    transverse = np.arange(width, dtype=np.float32) - (width - 1) / 2
    candidates = (np.arange(round(.13 * height), round(.31 * height)),
                  np.arange(round(.62 * height), round(.83 * height)))
    paths = []
    for rows in candidates:
        best = 0.
        chosen = None
        for slope in np.linspace(-.34, .34, 9):
            for bend in np.linspace(-.10, .26, 10):
                offset = slope * transverse + bend * width * (transverse / (width / 2)) ** 2
                trace = rows[:, None] + offset[columns]
                response = local[np.rint(trace).astype(np.int32).clip(0, height - 1), columns]
                support = (response >= 5).mean(1)
                clipped = np.minimum(response, 19)
                score = .5 * (clipped.mean(1) + np.percentile(clipped, 35, axis=1))
                score += 2 * support - .9 * abs(bend)
                score[support < .56] = 0
                winner = int(score.argmax())
                if score[winner] > best:
                    best, chosen = score[winner], rows[winner] + offset
        if best < 8 or chosen is None:
            return None
        paths.append(chosen)
    return np.array(paths, dtype=np.float32)


def wrapped_surfaces(rgb, mask, paths):
    """Fit diffuse colour planes between the banks and retain original shadows at them."""
    height, width = mask.shape
    y, x = np.indices(mask.shape, dtype=np.float32)
    x = (x + .5) / width * 2 - 1
    v = (y + .5) / height * 2 - 1
    basis = np.stack([np.ones(mask.shape, np.float32), x, x * x, v], axis=-1)
    first, last = y - paths[0], y - paths[1]
    interior = cv2.erode(mask, np.ones((5, 5), np.uint8)).astype(bool)
    margin = max(3, width * .062)
    cores = (first < -margin, (first > margin) & (last < -margin), last > margin)
    luminance = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    near_first = .5 + .5 * np.tanh(first / (width * .036))
    near_last = .5 + .5 * np.tanh(last / (width * .036))
    weights = (1 - near_first, near_first - near_last, near_last)
    plane = np.zeros_like(rgb)
    for core, weight in zip(cores, weights):
        use = interior & core
        if use.sum() < 30:
            return None
        low, high = np.percentile(luminance[use], (34, 91))
        sample = use & (luminance >= low) & (luminance <= high)
        fitted = np.linalg.lstsq(basis[sample], rgb[sample], rcond=None)[0]
        lower = np.percentile(rgb[use], 27, axis=0)
        upper = np.percentile(rgb[use], 93, axis=0)
        plane += np.clip(basis @ fitted, lower, upper) * weight[..., None]
    distance = np.minimum(np.abs(first), np.abs(last))
    bank = np.exp(-.5 * (distance / max(2., .055 * width)) ** 2)
    difference = rgb - plane
    shadow = cv2.cvtColor(difference, cv2.COLOR_RGB2GRAY) < 0
    retained = np.where(shadow, .09 + .86 * bank, .10 + .40 * bank)
    # Only broad source colour variation gets back into the cloth faces.
    # This leaves a soft turn of the cloth without retaining the small wedges.
    sigma = max(2., width * .13)
    core = mask.astype(np.float32)
    density = cv2.GaussianBlur(core, (0, 0), sigma)
    broad = cv2.GaussianBlur(difference * core[..., None], (0, 0), sigma)
    broad /= np.maximum(density[..., None], .01)
    rounded = np.clip(broad, -.035, .035) * (.32 * (1 - bank[..., None]))
    return plane + rounded + difference * retained[..., None], bank


def finish_cuffs(raw, stack, source, groups, materials, protect, strength):
    """Re-encode the edited wrap paint into diffuse, screen and straight ink layers."""
    if strength <= 0:
        return
    for roi, piece, matrix, size in limb_bands(groups, materials):
        distance = cv2.distanceTransform(piece.astype(np.uint8), cv2.DIST_L2, 5)
        opacity = strength * (1 - protect[roi]) * np.clip((distance - 2.) / 4., 0, 1)
        active = opacity > 0
        if not active.any():
            continue
        canonical = warp_into(piece.astype(np.uint8), matrix, size, nearest=True)
        paths = wrap_separators(warp_into(source[roi], matrix, size), canonical)
        if paths is None:
            continue
        diffuse = raw.flat[roi] * raw.shadow[roi]
        paint = diffuse + (1 - diffuse) * raw.highlight[roi]
        result = wrapped_surfaces(warp_into(paint, matrix, size), canonical, paths)
        if result is None:
            continue
        plane, banks = result
        height, width = piece.shape
        bank = cv2.warpAffine(banks, matrix, (width, height), flags=cv2.INTER_LINEAR)
        target = cv2.warpAffine(plane, matrix, (width, height), flags=cv2.INTER_LINEAR)
        edit = opacity[active, None]
        if stack.flat is raw.flat:
            stack.flat = stack.flat.copy()
        previous_flat = stack.flat[roi][active]
        now = previous_flat * stack.shadow[roi][active]
        # A narrow retained Screen component follows the existing supported bank.
        lift = (1 - diffuse[active]) * raw.highlight[roi][active] * (.28 * bank[active, None])
        diffuse_target = np.clip(target[active] - lift, 0, 1)
        mixed = now + edit * (diffuse_target - now)
        flat = np.maximum(previous_flat, mixed)
        stack.flat[roi][active] = flat
        stack.shadow[roi][active] = np.divide(mixed, flat, out=np.ones_like(flat), where=flat > 0)
        old_lift = (1 - now) * stack.highlight[roi][active]
        current_lift = old_lift + edit * (lift - old_lift)
        stack.highlight[roi][active] = np.clip(current_lift / np.maximum(1 - mixed, 1e-6), 0, 1)
        old_ink = stack.lines[roi][active]
        raw_ink = raw.lines[roi][active]
        existing = old_ink[:, 3:] * (1 - edit)
        original = raw_ink[:, 3:] * (.08 + .88 * bank[active, None]) * edit
        merged = existing + original
        ink = old_ink[:, :3] * existing + raw_ink[:, :3] * original
        stack.lines[roi][active, :3] = np.divide(ink, merged, out=np.zeros_like(ink), where=merged > 0)
        stack.lines[roi][active, 3:] = merged
