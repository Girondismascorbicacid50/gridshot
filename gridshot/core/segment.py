"""M1 segmentation: one-shot salient-object mask via IS-Net on ONNX Runtime.

The ChArUco mat makes a *busy* background, so classical thresholding is
hopeless — a learned saliency model is the floor.  ONNX Runtime is called
directly (rembg's dependency chain no longer installs on modern Python);
the model file is downloaded once into the config cache.  Inference runs on
a 1024² copy; the matte is resized back so geometry stays at full
resolution.  The interactive SAM segserver replaces this path in M3.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from .models import config_dir

MODEL_NAME = "isnet-general-use"
MODEL_URL = (
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
    "isnet-general-use.onnx"
)
INFER_SIDE = 1024

_session = None


def model_path() -> Path:
    override = os.environ.get("GRIDSHOT_SEG_MODEL")
    if override:
        return Path(override)
    return config_dir() / "cache" / f"{MODEL_NAME}.onnx"


def _ensure_model() -> Path:
    path = model_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)  # ~170MB, one-time
        tmp.rename(path)
    return path


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort

        _session = ort.InferenceSession(
            str(_ensure_model()), providers=["CPUExecutionProvider"]
        )
    return _session


def mask_for_image(pixels: np.ndarray) -> np.ndarray:
    """RGB HxWx3 uint8 → binary mask HxW uint8 (0/255) at full resolution."""
    h, w = pixels.shape[:2]
    small = np.asarray(
        Image.fromarray(pixels).resize((INFER_SIDE, INFER_SIDE), Image.LANCZOS),
        dtype=np.float32,
    )
    # IS-Net preprocessing (per rembg's DisSession): scale to [0,1], centre 0.5
    small = small / max(float(small.max()), 1e-6)
    small = (small - 0.5) / 1.0
    inp = small.transpose(2, 0, 1)[None].astype(np.float32)

    session = _get_session()
    input_name = session.get_inputs()[0].name
    pred = session.run(None, {input_name: inp})[0][0, 0]  # 1024x1024 float

    lo, hi = float(pred.min()), float(pred.max())
    if hi - lo > 1e-6:
        pred = (pred - lo) / (hi - lo)
    matte = (pred * 255).astype(np.uint8)
    matte = np.asarray(Image.fromarray(matte).resize((w, h), Image.BILINEAR))
    return ((matte > 127) * 255).astype(np.uint8)
