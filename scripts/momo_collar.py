"""Export the reviewed collar-shadow variation for Mo Mo's V-sign portrait."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from detail_protection import file_digest

SOURCE_RGBA_SHA256 = "67666a720bb5e47816efd9703c03217244533042745e57fc1649ffd7b0c06aa8"
COLLAR = np.array([(335, 353), (372, 349), (402, 339), (432, 371),
                   (432, 400), (440, 419), (450, 435), (436, 439),
                   (440, 454), (416, 442), (390, 423), (370, 404),
                   (352, 381)], dtype=np.int32)


def export(directory):
    report = json.loads((directory / "report.json").read_text())
    with Image.open(report["input"]) as im:
        source = np.asarray(im.convert("RGBA"))
    if source.shape != (1578, 997, 4) or hashlib.sha256(source.tobytes()).hexdigest() != SOURCE_RGBA_SHA256:
        raise ValueError("This local selection requires the reviewed V-sign portrait.")
    path = directory / "result.png"
    if file_digest(path) != report["detail_preservation"].get("result_sha256"):
        raise ValueError("Use the matching render from the daily pipeline.")
    with Image.open(path) as im:
        rendered = np.array(im.convert("RGBA"))
    if rendered.shape != source.shape or not np.array_equal(source[..., 3], rendered[..., 3]):
        raise ValueError("The render and original must have matching coverage.")

    x, y, width, height = cv2.boundingRect(COLLAR)
    # Leave one zero row and column around the polygon for its inward feather.
    x -= 1
    y -= 1
    area = np.s_[y:y+height+2, x:x+width+2]
    matte = np.zeros((height+2, width+2), np.uint8)
    cv2.fillPoly(matte, [COLLAR - np.array([x, y], np.int32)], 1)
    feather = np.clip(cv2.distanceTransform(matte, cv2.DIST_L2, 5) / 8, 0, 1)
    original = source[area].astype(np.float32)
    rgb = original[..., :3]
    paint = rendered[area][..., :3].astype(np.float32)
    blue = np.clip((rgb[..., 2] - rgb[..., 0] - 3) / 14, 0, 1)
    cloth = blue * np.clip((rgb[..., 0] - 104) / 28, 0, 1)
    opacity = np.clip((original[..., 3] - 200) / 50, 0, 1)
    gained_light = np.maximum((paint-rgb) @ np.array([.2126, .7152, .0722], np.float32), 0)
    shift = .55 * feather * cloth * gained_light * opacity
    rendered[area][..., :3] = np.clip(np.rint(paint - shift[..., None]), 0, 255).astype(np.uint8)
    output = directory / "result_momo_collar.png"
    Image.fromarray(rendered).save(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="daily output directory for the original portrait")
    args = parser.parse_args()
    try:
        print(export(args.directory))
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
