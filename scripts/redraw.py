#!/usr/bin/env python
"""Estimate and recompose an illustrator's layer stack."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import time
import warnings

import cv2
import numpy as np

from belt_finish import finish_belt_surfaces
from bands import soften_limb_bands
from accessories import crown_shells, perforated_straps
from colorfix import load_image, load_source, save_image, white_balance
from cuffs import finish_cuffs
import denoise_stage
import detail_protection
from layers import decompose, light_direction, rebuild_shadows, attach_reflections, cloth_ink, matte_fill_light
from masks import (line_mask, l0_smooth, segment_regions, region_stats, classify_regions,
                   group_materials, group_kinds, material_features, classify_material,
                   MATERIALS, MATERIAL_COLORS, protect_mask, overrender_map, face_feature_mask,
                   line_tiers, tier_debug, TIER_OUTLINE, TIER_INTER, TIER_INTRA, TIER_NAMES)
from suppress import detect_specular
from highlights import rebuild_highlights
from hair import finish_hair
from pastel import pastel_protection
from pigment import contour_coverage
from strokes import finish_lines
from scene import canvas_mode, analysis_plate, scene_protection
from details import (accessory_protection, attenuate_blush, detect_additional_faces, finish_accessories,
                     mouth_stroke_protection, profile_paint_protection, separate_accessories, whiten_eye_catchlights)

PRESETS = {"artist", "cel", "light"}


@dataclass
class Analysis:
    lines: np.ndarray
    labels: np.ndarray
    stats: dict
    kinds: dict
    sil: np.ndarray
    face: np.ndarray
    base_face: np.ndarray
    eyes: np.ndarray
    profiles: tuple
    gmap: np.ndarray
    materials: dict
    feats: dict
    spec: np.ndarray
    over: np.ndarray
    hardware: np.ndarray
    accessory_optics: object
    external_protect: np.ndarray | None
    protect: np.ndarray
    ink_protect: np.ndarray
    blush_protect: np.ndarray
    tiers: np.ndarray

    def decompose(self, img):
        return decompose(img, self.lines, self.gmap, self.materials, self.tiers == TIER_INTRA)


def overlay_heat(img, heat):
    h = np.uint8(np.clip(heat * 255, 0, 255))
    cm = cv2.applyColorMap(h, cv2.COLORMAP_INFERNO)[..., ::-1].astype(np.float32) / 255
    return img * 0.45 + cm * 0.55


def analyze(img, mode, alpha, override, strength, protect_eyes):
    cv2.setRNGSeed(0)
    lines = line_mask(img)
    bg = None
    if mode == "transparent":
        bg = alpha == 0
        lines &= ~bg
    structure = l0_smooth(img)
    labels = segment_regions(structure, lines, raw=img, bg=bg)
    stats = region_stats(structure, labels, lines)
    kinds = classify_regions(structure, labels, lines, stats)
    groups = group_materials(structure, labels, kinds, stats)
    for index, parts in enumerate(crown_shells(labels, stats), int(labels.max()) + 1):
        groups[parts] = index
        kinds.update(dict.fromkeys(parts, "horn"))
    gkinds = group_kinds(groups, kinds, stats)
    gmap, gkinds, hardware = perforated_straps(img, labels, groups[labels], gkinds)
    over = overrender_map(img, lines, structure)
    spec = detect_specular(img, gmap, gkinds, lines)
    feats = material_features(img, gmap, lines, spec, over) if gmap.any() else {}
    materials = classify_material(img, gmap, lines, gkinds, spec, over, feats)
    gmap, reflections = attach_reflections(img, gmap, materials, lines)
    sil = labels > 0
    face = face_feature_mask(labels, kinds, stats, sil)
    eyes = np.isin(labels, [r for r in kinds if kinds[r] == "eyes"])
    profiles = ()
    base_face = face
    if strength > 0:
        face, eyes, profiles = detect_additional_faces(img, labels, stats, lines, face, eyes)
    accessory_optics = None
    if override is None:
        protect = protect_mask(img, labels, kinds, stats, eyes=protect_eyes, eye_level=1,
                               reflections=reflections)
        if protect_eyes:
            protect[eyes] = 1
        protect = np.maximum(protect, hardware)
        if strength > 0:
            accessories, accessory_optics = accessory_protection(
                img, labels, kinds, gmap, materials, lines, reflections)
            protect = np.maximum(protect, accessories)
            gmap, materials = separate_accessories(gmap, materials, accessory_optics)
            feats = material_features(img, gmap, lines, spec, over) if gmap.any() else {}
    else:
        protect = override
    external_protect = None
    if mode == "scene" and override is None:
        external_protect = scene_protection(img, labels, lines)
    elif mode == "transparent":
        external_protect = 1 - alpha.astype(np.float32) / 255
    if external_protect is not None:
        protect = np.maximum(protect, external_protect)
    if override is None and strength > 0:
        protect = np.maximum(protect, mouth_stroke_protection(img, labels, kinds, stats, base_face, eyes))
        pastel = pastel_protection(img, sil, lines, alpha)
        if pastel is not None:
            protect = np.maximum(protect, pastel)
    tiers = line_tiers(lines, sil, face, gmap)
    ink_protect = protect
    protect = np.maximum(protect,contour_coverage(img,sil,tiers==TIER_OUTLINE))
    blush_protect = protect
    if profiles and override is None:
        protect = np.maximum(protect, profile_paint_protection(img, sil, profiles))
    return Analysis(lines, labels, stats, kinds, sil, face, base_face, eyes, profiles, gmap, materials,
                    feats, spec, over, hardware, accessory_optics, external_protect, protect,
                    ink_protect, blush_protect, tiers)


def run(path, out_dir=None, preset="artist", strength=None, budget=None,
        highlight=None, blush=None, protect_eyes=True, mask=None, policy=None,
        inner_line_strength=None, verbose=True, return_debug=False, *, denoise="off",
        denoise_amount=denoise_stage.DEFAULT_AMOUNT, denoise_models=None, denoise_device="auto",
        denoise_python=None, detail_protect="auto", detail_mask=None, auto_skin_ridges=None, protect_repeated_print=None,
        protect_metal_links=None):
    start = time.perf_counter()
    if preset not in PRESETS:
        raise ValueError("unknown preset " + preset)
    preset = "artist" if preset == "cel" else preset
    if budget:
        warnings.warn("--budget is obsolete for continuous-tone layers", UserWarning)
    if policy:
        raise ValueError("cel material policy overrides are obsolete for artist layers")
    daily = preset == "artist"
    if auto_skin_ridges is None:
        auto_skin_ridges = daily
    if protect_repeated_print is None:
        protect_repeated_print = daily
    if protect_metal_links is None:
        protect_metal_links = daily
    strength = (0.45 if preset == "light" else 1.0) if strength is None else float(strength)
    if not 0 <= strength <= 1:
        raise ValueError("strength must be in [0,1]")
    if detail_protect not in ("auto", "off") or (detail_protect == "off" and detail_mask is not None):
        raise ValueError("choose auto or off for detail protection; a detail mask requires auto")
    denoise_info = denoise_stage.settings(denoise, denoise_amount, strength, denoise_models,
                                          denoise_device, denoise_python)
    highlight = highlight or ("keep" if preset=="light" else "artist")
    blush = blush or ("keep" if preset == "light" else "attenuate")
    if highlight not in ("keep", "artist") or blush not in ("keep", "attenuate"):
        raise ValueError("highlight supports artist and keep; blush supports attenuate and keep")
    directory = Path(out_dir or (str(Path(path).with_suffix("")) + "_deai"))
    directory.mkdir(parents=True, exist_ok=True)
    src, alpha = load_source(path)
    detail_weight = None
    previous_weight = None
    detail_mode = "off"
    if detail_mask is not None:
        detail_weight = detail_protection.read_mask(detail_mask, src.shape[:2])
        detail_mode = "explicit"
    elif detail_protect == "auto" and strength > 0 and mask is None:
        detail_weight = detail_protection.reviewed_mask(src, alpha)
        detail_mode = "reviewed" if detail_weight is not None else "automatic"
        if detail_weight is None:
            previous_weight = detail_protection.previous_result_mask(path, src.shape[:2])
    mode = canvas_mode(src, alpha)
    source_plate = analysis_plate(src, alpha) if mode == "transparent" else src
    override = None if mask is None else cv2.resize(load_image(mask).mean(2), src.shape[1::-1])
    if mode == "paper":
        img, wb = white_balance(source_plate)
        if wb["applied"]:
            exposure = strength if override is None else strength * (1 - override[..., None])
            img = source_plate + exposure * (img - source_plate)
    else:
        img, wb = source_plate, {"applied": False, "reason": "no neutral backdrop"}

    unfiltered_plate = img
    analysis = analyze(unfiltered_plate, mode, alpha, override, strength, protect_eyes)
    img, denoise_weight = denoise_stage.apply(unfiltered_plate, analysis.protect, denoise_info)
    if denoise_info["applied"]:
        denoise_info["analysis_mode"] = "source_geometry"
    raw = analysis.decompose(img)
    # Keep light direction and confidence tied to the original R5 analysis plate.
    light_stack = analysis.decompose(unfiltered_plate) if denoise_info["applied"] else raw
    light, confidence = light_direction(light_stack, analysis.gmap)
    source_stack = analysis.decompose(source_plate) if wb["applied"] else light_stack
    source_stack.write(directory / "original_layers", alpha, src)
    # The automatic comparison uses white-balanced source paint to avoid
    # reintroducing a paper colour cast. Reviewed and manual masks specify
    # literal source colours. All references precede denoising.
    detail_source = None
    if detail_mode == "automatic":
        detail_source = light_stack if previous_weight is None else source_stack
    elif detail_weight is not None:
        detail_source = source_stack
    del light_stack
    hard_protect = override >= 1 if denoise_info["applied"] and override is not None else None
    if hard_protect is None or not hard_protect.any():
        hard_protect = None
        del source_stack

    lines, labels, stats, kinds = analysis.lines, analysis.labels, analysis.stats, analysis.kinds
    sil, face, base_face, eyes, profiles = analysis.sil, analysis.face, analysis.base_face, analysis.eyes, analysis.profiles
    gmap, materials, feats = analysis.gmap, analysis.materials, analysis.feats
    spec, over, hardware = analysis.spec, analysis.over, analysis.hardware
    accessory_optics, external_protect = analysis.accessory_optics, analysis.external_protect
    protect, ink_protect, blush_protect, tiers = (analysis.protect, analysis.ink_protect,
                                                 analysis.blush_protect, analysis.tiers)
    del analysis
    stack = replace(raw, shadow=rebuild_shadows(raw, gmap, materials, protect, strength, light, confidence))
    stack.highlight = matte_fill_light(raw, stack.shadow, gmap, materials, protect, strength)
    stack.lines = cloth_ink(raw, stack.shadow, gmap, materials, tiers, protect, strength)
    if highlight=="artist" and strength>0:
        stack.shadow, stack.highlight, stack.lines = rebuild_highlights(raw, stack, img, gmap, materials, protect, strength, light)
        if accessory_optics is not None:
            finish_accessories(raw, stack, accessory_optics, strength, light, external_protect)
        soften_limb_bands(stack, gmap, materials, protect, strength)
    if preset=="artist" and strength>0:
        stack.lines = finish_lines(stack,img,tiers,gmap,materials,face,sil,ink_protect,strength,light)
        finish_hair(raw, stack, img, gmap, materials, tiers, sil, face, protect,
                    strength, light, edit_highlights=highlight == "artist",
                    scene_protect=external_protect if mode == "scene" and override is None else None)
    if highlight == "artist" and strength > 0:
        belt_override = ink_protect if override is not None else external_protect
        finish_belt_surfaces(raw, stack, img, gmap, materials, hardware, belt_override,
                             strength, light, accessory_optics)
        if preset == "artist":
            finish_cuffs(raw, stack, img, gmap, materials, protect, strength)
    if inner_line_strength is not None:
        ink = stack.lines.copy()
        for tier, opacity in ((TIER_INTRA, inner_line_strength), (TIER_INTER, inner_line_strength + .3)):
            active = tiers == tier
            ink[..., 3][active] *= 1 - strength * (1-protect[active]) * (1-np.clip(opacity, 0, 1))
        stack.lines = ink
    if strength > 0:
        if blush == "attenuate":
            attenuate_blush(stack, img, labels, kinds, stats, base_face, eyes, blush_protect, strength, profiles)
        if highlight == "artist" and protect_eyes and mask is None:
            whiten_eye_catchlights(img, stack, eyes, strength, external_protect)
    if hard_protect is not None:
        # Rendering can use neighbouring colours. Keep explicit hard protection through the layer stack.
        stack.flat[hard_protect] = source_stack.flat[hard_protect]
        stack.shadow[hard_protect] = source_stack.shadow[hard_protect]
        stack.highlight[hard_protect] = source_stack.highlight[hard_protect]
        stack.lines[hard_protect] = source_stack.lines[hard_protect]
    print_pixels = 0
    metal_link_pixels = 0
    if detail_mode == "automatic":
        provisional = stack.composite()
        detail_weight = detail_protection.automatic_mask(unfiltered_plate, provisional,
                                                        gmap, materials, alpha)
        # Card PNGs sometimes have a lightly translucent border. Reserve this
        # optional repair for filled canvases, excluding isolated character cutouts.
        if protect_repeated_print and (mode == "scene" or
                (mode == "transparent" and np.mean(alpha < 128) < .02)):
            import repeated_print
            pattern_weight = repeated_print.protection(unfiltered_plate, provisional, alpha)
            print_pixels = int(np.count_nonzero(pattern_weight))
            np.maximum(detail_weight, pattern_weight, out=detail_weight)
        if previous_weight is not None:
            np.maximum(detail_weight, previous_weight, out=detail_weight)
        if protect_metal_links:
            import metal_links
            metal_weight = metal_links.protection(unfiltered_plate, provisional, alpha, detail_weight)
            metal_link_pixels = int(np.count_nonzero(metal_weight))
            np.maximum(detail_weight, metal_weight, out=detail_weight)
        del provisional
    if detail_weight is not None:
        if alpha is not None:
            detail_weight[alpha == 0] = 0
        if detail_weight.any():
            original = detail_protection.snapshot(detail_source, detail_weight > 0)
            detail_protection.restore(stack, original, detail_weight)
    del detail_source, unfiltered_plate
    ridge_info = None
    if auto_skin_ridges and strength > 0 and highlight == "artist" and mask is None:
        import auto_ridges
        ridge_info = auto_ridges.apply(stack, src, alpha, detail_weight, strength)
    stack.write(directory, alpha, src)
    result = stack.composite()
    if alpha is None:
        save_image(directory / "result.png", result)
    else:
        result = np.where((alpha == 0)[..., None], src, result)
        save_image(directory / "result.png", np.dstack((result, alpha.astype(np.float32) / 255)))
    rng = np.random.default_rng(7)
    pal = rng.uniform(0.15, 1, (int(labels.max()) + 1, 3)).astype(np.float32)
    pal[0] = 1
    dr = pal[labels]
    dr[lines] = 0
    save_image(directory / "debug_regions.png", dr)
    save_image(directory / "debug_lines.png", (~lines).astype(np.float32))
    save_image(directory / "debug_overrender.png", overlay_heat(img, over))
    save_image(directory / "debug_protect.png", protect)
    save_image(directory / "debug_detail_protect.png", detail_weight if detail_weight is not None
               else np.zeros_like(protect))
    save_image(directory / "debug_linetiers.png", tier_debug(img, tiers))
    save_image(directory / "debug_materials.png", material_debug(gmap, materials, lines, spec))
    report = {"checkpoint": "R5 selective highlights and structural folds", "input": str(Path(path).resolve()), "size": list(img.shape[1::-1]),
              "params": {"preset": preset, "strength": strength, "highlight": highlight,
                         "blush": blush, "protect_eyes": protect_eyes},
              "light_to_source_xy": light.tolist(), "light_confidence": confidence,
              "white_balance": wb, "denoise": denoise_info, "canvas": mode, "source_alpha": alpha is not None, "regions": [
                  {"group": int(g), "material": materials[g], "features": f} for g, f in feats.items()],
              "detail_preservation": {"mode": detail_mode,
                                       "carried_from_input": previous_weight is not None,
                                       "active_pixels": int(np.count_nonzero(detail_weight))},
              "seconds": round(time.perf_counter() - start, 2)}
    if detail_weight is not None and detail_weight.any():
        info = report["detail_preservation"]
        info["result_sha256"] = detail_protection.file_digest(directory / "result.png")
        info["matte_sha256"] = detail_protection.file_digest(directory / "debug_detail_protect.png")
    report["params"].update(auto_skin_ridges=auto_skin_ridges,
                            protect_repeated_print=protect_repeated_print,
                            protect_metal_links=protect_metal_links)
    if protect_metal_links:
        report["metal_link_pixels"] = metal_link_pixels
    if protect_repeated_print:
        report["repeated_print_pixels"] = print_pixels
    if ridge_info is not None:
        report["skin_ridges"] = ridge_info
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if verbose:
        print("done in %.2fs -> %s" % (report["seconds"], directory))
    if return_debug:
        return result, report, {"img": img, "lines": lines, "silhouette": sil, "face": face,
                                "tiers": tiers, "gmap": gmap, "materials": materials,
                                "accessories": accessory_optics, "eyes": eyes, "profiles": profiles,
                                "denoise_weight": denoise_weight}
    return result, report


def material_debug(gmap, materials, lines, spec):
    """Colour-coded material map with a legend strip at the top."""
    H, W = gmap.shape
    # Reflections may exhaust a high-numbered group during reassignment.
    lut = np.ones((max(int(gmap.max()), max(materials, default=0)) + 1, 3), np.float32)
    for g, m in materials.items():
        lut[g] = MATERIAL_COLORS.get(m, (0.5, 0.5, 0.5))
    vis = lut[gmap]
    vis[spec] = vis[spec] * 0.4 + 0.6
    vis[lines] = 0.0
    names = [m for m in MATERIALS if m != "background"]
    cell = W // len(names)
    bar = np.ones((44, W, 3), np.float32)
    for i, m in enumerate(names):
        bar[6:38, i * cell + 4:i * cell + 30] = MATERIAL_COLORS[m]
        u8 = np.ascontiguousarray((bar * 255).astype(np.uint8))
        cv2.putText(u8, m, (i * cell + 34, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
                    cv2.LINE_AA)
        bar = u8.astype(np.float32) / 255
    return np.concatenate([bar, vis], 0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--out", dest="out_dir")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="artist")
    ap.add_argument("--strength", type=float)
    ap.add_argument("--budget", help="deprecated cel-only option")
    ap.add_argument("--highlight", choices=["artist","keep"])
    ap.add_argument("--blush", choices=["attenuate", "keep"])
    ap.add_argument("--protect-eyes", dest="protect_eyes", action="store_true", default=True)
    ap.add_argument("--no-protect-eyes", dest="protect_eyes", action="store_false")
    ap.add_argument("--policy", help="legacy cel-only option (unsupported)")
    ap.add_argument("--inner-line-strength", type=float, dest="inner_line",
                    help="0..1 strength of lines inside one material (folds, pleats); "
                         "material-border lines get +0.3; outline and face stay 1.0")
    ap.add_argument("--mask", help="protection override PNG: white = keep original, "
                                   "black = process fully (replaces the automatic protect mask)")
    ap.add_argument("--denoise", choices=denoise_stage.METHODS, default="off",
                    help="v1.5.0 preprocessing method (default: off)")
    ap.add_argument("--denoise-amount", type=float, default=denoise_stage.DEFAULT_AMOUNT,
                    help="0..1 blend before strength and protection (default: %(default)s)")
    ap.add_argument("--denoise-models", help="local directory for SHA-256 verified ESRGAN weights")
    ap.add_argument("--denoise-device", choices=denoise_stage.DEVICES, default="auto")
    ap.add_argument("--denoise-python", help="Python with the optional upstream dependencies")
    ap.add_argument("--detail-protect", choices=("auto", "off"), default="auto",
                    help="retain lost ornament detail by default")
    ap.add_argument("--detail-mask", help="source-sized grayscale PNG replacing automatic detail selection; white retains source")
    ap.add_argument("--auto-skin-ridges", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--protect-repeated-print", action=argparse.BooleanOptionalAction, default=None, help="preserve recurring motifs in full-scene cards with automatic detail protection")
    ap.add_argument("--protect-metal-links", action=argparse.BooleanOptionalAction, default=None, help="keep lost small metal glints and curved rims in automatic detail mode")
    a = ap.parse_args(argv)
    if a.strength is not None and not 0 <= a.strength <= 1:
        ap.error("--strength must be in 0..1")
    if not 0 <= a.denoise_amount <= 1:
        ap.error("--denoise-amount must be in 0..1")
    if a.detail_mask and a.detail_protect == "off":
        ap.error("--detail-mask requires --detail-protect auto")
    try:
        run(a.input, a.out_dir, a.preset, a.strength, a.budget, a.highlight, a.blush,
            a.protect_eyes, a.mask, a.policy, a.inner_line, denoise=a.denoise,
            denoise_amount=a.denoise_amount, denoise_models=a.denoise_models,
            denoise_device=a.denoise_device, denoise_python=a.denoise_python,
            detail_protect=a.detail_protect, detail_mask=a.detail_mask, auto_skin_ridges=a.auto_skin_ridges,
            protect_repeated_print=a.protect_repeated_print, protect_metal_links=a.protect_metal_links)
    except denoise_stage.DenoiseError as exc:
        ap.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
