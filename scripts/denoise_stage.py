"""Optional v1.5.0 denoising of an opaque R5 analysis plate."""

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image


METHODS = ("off", "esrgan60_l0", "l0_0.001", "l0_0.0005")
DEVICES = ("auto", "cpu", "mps", "cuda")
DEFAULT_AMOUNT = 0.30
KERNEL_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS = KERNEL_DIR / "vendor" / "gpt_anime_denoise" / "models"


class DenoiseError(RuntimeError):
    """The optional denoising stage could not finish."""


def settings(method, amount, strength, models_dir, device, python):
    amount = float(amount)
    if method not in METHODS:
        raise ValueError(f"unknown denoise method {method!r}")
    if device not in DEVICES:
        raise ValueError(f"unknown denoise device {device!r}")
    if not 0 <= amount <= 1:
        raise ValueError("denoise amount must be in [0,1]")
    model_folder = Path(models_dir).expanduser().resolve() if models_dir else DEFAULT_MODELS
    venv = KERNEL_DIR.parent.parent / ".denoise-venv"
    local_python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python = os.path.expanduser(python) if python else str(local_python if local_python.is_file() else sys.executable)
    reason = None
    if method == "off":
        reason = "disabled"
    elif strength == 0:
        reason = "zero_strength"
    elif amount == 0:
        reason = "zero_amount"
    return {
        "upstream_version": "v1.5.0",
        "method": method,
        "amount": amount,
        "strength_scaled_amount": amount * strength if method != "off" else 0.0,
        "effective_max_amount": 0.0,
        "effective_mean_amount": 0.0,
        "device_requested": device,
        "device": None,
        "models_dir": str(model_folder),
        "python": python,
        "applied": False,
        "reason": reason,
    }


def apply(plate, protect, info):
    if info["reason"] is not None:
        return plate, None

    weight = np.float32(info["strength_scaled_amount"]) * (1 - np.clip(protect, 0, 1))
    if not weight.any():
        info["reason"] = "fully_protected"
        return plate, weight

    with tempfile.TemporaryDirectory(prefix="r5-denoise-") as temp:
        source = Path(temp) / "input.png"
        filtered = Path(temp) / "filtered.png"
        pixels = np.rint(np.clip(plate, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(pixels).save(source)
        command = [info["python"], "-B", str(Path(__file__).resolve()), str(source), str(filtered),
                   info["method"], info["device_requested"], info["models_dir"]]
        try:
            worker = subprocess.run(command, check=True, capture_output=True, text=True)
            metadata = json.loads(worker.stdout)
            with Image.open(filtered) as image:
                clean = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        except subprocess.CalledProcessError as exc:
            details = exc.stderr.strip() or exc.stdout.strip() or f"exit code {exc.returncode}"
            raise DenoiseError(f"denoising with {info['method']} failed:\n{details}") from exc
        except (OSError, ValueError) as exc:
            raise DenoiseError(f"could not run the denoise worker with {info['python']}: {exc}") from exc
    if clean.shape != plate.shape:
        raise DenoiseError(f"expected denoised shape {plate.shape}, got {clean.shape}")

    mixed = np.clip(plate + (clean - plate) * weight[..., None], 0, 1)
    info.update(metadata, effective_max_amount=float(weight.max()),
                effective_mean_amount=float(weight.mean(dtype=np.float64)), applied=True)
    return mixed, weight


def _run_worker():
    # Import model dependencies only in the optional worker interpreter.
    from vendor.gpt_anime_denoise import denoise as upstream

    source, output, method, requested_device, models_dir = sys.argv[1:]
    device = upstream.default_device() if requested_device == "auto" else requested_device
    models = {}
    metadata = {"device": device}
    if method == "esrgan60_l0":
        name = upstream.MODELS["esrgan"][0]
        model_path = Path(models_dir) / name
        if not model_path.is_file():
            raise SystemExit(f"Local model required at {model_path}. {upstream.manual_hint('esrgan', models_dir)}")
        data = model_path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != upstream.SHA256[name]:
            raise SystemExit(f"SHA-256 mismatch for {model_path}. Expected {upstream.SHA256[name]}, got {digest}.")

        import spandrel

        weights = upstream.torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        models["esrgan"] = spandrel.ModelLoader().load_from_state_dict(weights).model.to(device).eval()
        metadata["model_sha256"] = digest

    with Image.open(source) as image:
        result = upstream.run(image.convert("RGB"), [method], models, device)[method]
    result.save(output, "PNG")
    print(json.dumps(metadata))


if __name__ == "__main__":
    _run_worker()
