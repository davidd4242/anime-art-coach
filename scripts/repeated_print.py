"""Keep recurring small shapes where a provisional redraw loses local contrast."""
import cv2
import numpy as np


def protection(source, render, alpha, scale=None):
    height, width = source.shape[:2]
    scale = min(height, width) / 1024 if scale is None else scale
    half = max(5, round(7 * scale))
    reach = 4 * half

    def band(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray - cv2.GaussianBlur(gray, (0, 0), max(.85, 2.4 * scale))

    before = band(source)
    after = band(render)
    lost = (np.abs(before) > .026) & (before * after < .67 * before**2)
    if alpha is not None:
        lost &= alpha > 220
    size = 2 * half + 1
    coverage = cv2.boxFilter(lost.astype('float32'), -1, (size, size))
    stride = max(3, half)
    offset = half + reach
    interior = coverage[offset:height-offset:stride, offset:width-offset:stride]
    protected = np.zeros((height, width), 'float32')
    yy, xx = np.mgrid[-reach:reach+1, -reach:reach+1]
    distance_sq = xx*xx + yy*yy
    search = (distance_sq > (1.65 * half)**2) & (distance_sq <= reach**2)
    for grid_y, grid_x in np.argwhere(interior > .055):
        x = int(grid_x) * stride + offset
        y = int(grid_y) * stride + offset
        tile = before[y-half:y+half+1, x-half:x+half+1]
        # A corner test keeps a single straight contour from looking like a print.
        if cv2.cornerMinEigenVal(tile, 3).max() < .0004:
            continue
        area = before[y-offset:y+offset+1, x-offset:x+offset+1]
        matches = cv2.matchTemplate(area, tile, cv2.TM_CCOEFF_NORMED)
        matches[~search] = -1
        row, column = np.unravel_index(int(matches.argmax()), matches.shape)
        if matches[row, column] < .66:
            continue
        dy, dx = row - reach, column - reach
        # Require a third occurrence near the opposite offset, allowing for curvature.
        other_y, other_x = reach - dy, reach - dx
        tolerance = max(2, round(half * .75))
        opposite = matches[max(0, other_y-tolerance):other_y+tolerance+1,
                           max(0, other_x-tolerance):other_x+tolerance+1]
        if opposite.max() < .52:
            continue
        for center in [(x, y), (x+dx, y+dy), (x-dx, y-dy)]:
            cv2.circle(protected, center, round(1.7 * half), 1., -1)
    if protected.any():
        protected = cv2.GaussianBlur(protected, (0, 0), max(1., scale * 1.5))
        protected[protected < .01] = 0
    if alpha is not None:
        protected[alpha == 0] = 0
    return protected
