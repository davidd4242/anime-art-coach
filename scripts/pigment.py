"""Measure stroke profiles and preserve local pigment when light changes."""
import cv2
import numpy as np

from colorfix import to_lab, from_lab


def paired_banks(value, radius):
    """Depth of a valley with two lighter banks, sampled in several normals.
    A monotone boundary between two illuminated planes has no such valley."""
    pad = cv2.copyMakeBorder(value, radius, radius, radius, radius, cv2.BORDER_REPLICATE)
    height, width = value.shape
    depth = np.zeros_like(value)
    for angle in np.arange(4) * np.pi / 4:
        x, y = np.rint(radius * np.array([np.cos(angle), np.sin(angle)])).astype(int)
        a = pad[radius+y:radius+y+height, radius+x:radius+x+width]
        b = pad[radius-y:radius-y+height, radius-x:radius-x+width]
        np.maximum(depth, np.minimum(a,b)-value, out=depth)
    return depth


def contour_coverage(img, silhouette, dark_contour):
    """Hold original contour coverage and high-key lit rims. Grow no ink.
    Limit the rim term to an actual bright crest, so a white sleeve does not
    automatically acquire a white outline."""
    value = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    inner = cv2.distanceTransform(silhouette.astype(np.uint8),cv2.DIST_L2,5)
    dark = dark_contour.astype(np.float32)
    support = cv2.GaussianBlur(dark,(0,0),.8)
    # Use the subject side only. White background on the opposite bank
    # should not erase a real bright edge under directional illumination.
    weights = silhouette.astype(np.float32)
    body = cv2.GaussianBlur(value*weights,(0,0),3)/np.maximum(cv2.GaussianBlur(weights,(0,0),3),1e-5)
    bright = np.clip((value-body-.04)/.14,0,1)*np.clip(1-inner/5,0,1)
    return np.maximum(np.maximum(dark,support),bright*weights)


def colour_support(rgb, mask, sigma):
    """Local colour that survives a small spatial average, independent of L*.
    Neutral shading, white reflections, and single-pixel noise get low weight."""
    lab = to_lab(rgb)
    weights = mask.astype(np.float32)
    def mean(field, scale):
        den = cv2.GaussianBlur(weights,(0,0),scale)
        return cv2.GaussianBlur(field*weights[...,None],(0,0),scale)/np.maximum(den[...,None],1e-5)
    ab = lab[...,1:].copy()
    fine = mean(ab, .85)
    broad = mean(ab, sigma)
    excursion = np.linalg.norm(fine-broad,axis=2)
    noise = np.linalg.norm(ab-fine,axis=2)
    strength = np.clip((excursion-noise-1.5)/8,0,1)
    return strength*np.clip((np.linalg.norm(ab,axis=2)-3)/10,0,1)*weights


def restore_chroma(edited, original_lab, coverage):
    """Move two opponent channels towards the source at the edited lightness."""
    lab = to_lab(edited)
    lab[...,1:] += coverage[...,None]*(original_lab[...,1:]-lab[...,1:])
    return np.where(coverage[...,None]>0, from_lab(lab), edited)
