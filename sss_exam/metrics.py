"""Evaluation metrics commonly reported for image secret sharing."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


def entropy(arr: np.ndarray) -> float:
    """Shannon entropy in bits per byte (8.0 = ideal random)."""
    counts = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def adjacent_correlation(img: np.ndarray, samples: int = 20000) -> float:
    """Correlation between horizontally adjacent pixels (first channel)."""
    ch = img[..., 0] if img.ndim == 3 else img
    H, W = ch.shape
    if W < 2:
        return 0.0
    rng = np.random.default_rng(0)
    ys = rng.integers(0, H, samples)
    xs = rng.integers(0, W - 1, samples)
    a = ch[ys, xs].astype(np.float64)
    b = ch[ys, xs + 1].astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def npcr_uaci(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """NPCR (% pixels that differ) and UACI (% mean intensity difference)."""
    a = a.astype(np.int16)
    b = b.astype(np.int16)
    npcr = float((a != b).mean() * 100)
    uaci = float((np.abs(a - b) / 255).mean() * 100)
    return npcr, uaci


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def png_size(img: np.ndarray) -> int:
    buf = io.BytesIO()
    arr = img[..., 0] if img.ndim == 3 and img.shape[2] == 1 else img
    Image.fromarray(arr).save(buf, "PNG", optimize=False)
    return buf.tell()
