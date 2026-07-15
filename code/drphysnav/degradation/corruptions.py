"""13-type visual degradation benchmark (ImageNet-C style; PANav Sec. 4.1).

Each corruption takes an RGB image (HWC uint8 or float32 in [0,1]) and a
severity in {1,2,3,4}, returning the corrupted image in the same format.
Used for (a) degradation-augmented Student distillation and (b) the systematic
robustness evaluation. Severity tables are FROZEN before evaluation.

13 types:
    sensor : gaussian_noise, salt_pepper, motion_blur, low_light, fog, jpeg
    color  : color_jitter, red_filter, blue_filter, green_filter,
             warm_filter, cool_filter
    compose: combined
"""

from __future__ import annotations

import io
from typing import Callable, Dict, List

import cv2
import numpy as np
from PIL import Image

_GAUSS_STD = [0.04, 0.10, 0.20, 0.35]
_SALT_AMOUNT = [0.02, 0.06, 0.12, 0.22]
_MOTION_KSIZE = [5, 9, 15, 25]
_LOW_LIGHT_GAMMA = [1.6, 2.4, 3.4, 4.8]
_FOG_INTENSITY = [0.2, 0.4, 0.6, 0.8]
_JPEG_QUALITY = [50, 30, 18, 8]
_JITTER_SCALE = [0.2, 0.4, 0.6, 0.8]
_FILTER_STRENGTH = [0.15, 0.30, 0.45, 0.65]


def _to_float_hwc(img: np.ndarray):
    was_uint8 = img.dtype == np.uint8
    x = img.astype(np.float32)
    if was_uint8:
        x = x / 255.0
    return x, was_uint8


def _restore(x: np.ndarray, was_uint8: bool) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    if was_uint8:
        return (x * 255.0).round().astype(np.uint8)
    return x.astype(np.float32)


def _sev_idx(severity: int) -> int:
    return int(np.clip(severity, 1, 4)) - 1


def gaussian_noise(img, severity=1):
    x, u8 = _to_float_hwc(img)
    std = _GAUSS_STD[_sev_idx(severity)]
    x = x + np.random.normal(0.0, std, x.shape).astype(np.float32)
    return _restore(x, u8)


def salt_pepper(img, severity=1):
    x, u8 = _to_float_hwc(img)
    amount = _SALT_AMOUNT[_sev_idx(severity)]
    mask = np.random.rand(*x.shape[:2])
    x[mask < amount / 2] = 0.0
    x[mask > 1 - amount / 2] = 1.0
    return _restore(x, u8)


def motion_blur(img, severity=1):
    x, u8 = _to_float_hwc(img)
    k = _MOTION_KSIZE[_sev_idx(severity)]
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    x = cv2.filter2D(x, -1, kernel)
    return _restore(x, u8)


def low_light(img, severity=1):
    x, u8 = _to_float_hwc(img)
    gamma = _LOW_LIGHT_GAMMA[_sev_idx(severity)]
    x = np.power(x, gamma)
    return _restore(x, u8)


def fog(img, severity=1):
    x, u8 = _to_float_hwc(img)
    t = _FOG_INTENSITY[_sev_idx(severity)]
    x = x * (1 - t) + 0.9 * t
    return _restore(x, u8)


def jpeg_compression(img, severity=1):
    x, u8 = _to_float_hwc(img)
    q = _JPEG_QUALITY[_sev_idx(severity)]
    pil = Image.fromarray((x * 255).astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    out = np.asarray(Image.open(buf)).astype(np.float32) / 255.0
    return _restore(out, u8)


def color_jitter(img, severity=1):
    x, u8 = _to_float_hwc(img)
    s = _JITTER_SCALE[_sev_idx(severity)]
    bright = 1.0 + np.random.uniform(-s, s)
    contrast = 1.0 + np.random.uniform(-s, s)
    x = x * bright
    mean = x.mean(axis=(0, 1), keepdims=True)
    x = (x - mean) * contrast + mean
    return _restore(x, u8)


def _channel_filter(img, severity, channel: int):
    x, u8 = _to_float_hwc(img)
    strength = _FILTER_STRENGTH[_sev_idx(severity)]
    x[..., channel] = x[..., channel] * (1 + strength)
    for c in range(3):
        if c != channel:
            x[..., c] = x[..., c] * (1 - strength * 0.5)
    return _restore(x, u8)


def red_filter(img, severity=1):
    return _channel_filter(img, severity, 0)


def green_filter(img, severity=1):
    return _channel_filter(img, severity, 1)


def blue_filter(img, severity=1):
    return _channel_filter(img, severity, 2)


def warm_filter(img, severity=1):
    x, u8 = _to_float_hwc(img)
    s = _FILTER_STRENGTH[_sev_idx(severity)]
    x[..., 0] *= 1 + s
    x[..., 2] *= 1 - s * 0.7
    return _restore(x, u8)


def cool_filter(img, severity=1):
    x, u8 = _to_float_hwc(img)
    s = _FILTER_STRENGTH[_sev_idx(severity)]
    x[..., 2] *= 1 + s
    x[..., 0] *= 1 - s * 0.7
    return _restore(x, u8)


def combined(img, severity=1):
    sub = max(1, severity - 1)
    out = low_light(img, sub)
    out = gaussian_noise(out, sub)
    out = motion_blur(out, sub)
    return out


SENSOR_TYPES: List[str] = [
    "gaussian_noise", "salt_pepper", "motion_blur", "low_light", "fog", "jpeg",
]
COLOR_TYPES: List[str] = [
    "color_jitter", "red_filter", "blue_filter", "green_filter",
    "warm_filter", "cool_filter",
]
COMPOSITE_TYPES: List[str] = ["combined"]
AUGMENT_TYPES: List[str] = SENSOR_TYPES + COLOR_TYPES
ALL_TYPES: List[str] = AUGMENT_TYPES + COMPOSITE_TYPES

REGISTRY: Dict[str, Callable] = {
    "gaussian_noise": gaussian_noise, "salt_pepper": salt_pepper,
    "motion_blur": motion_blur, "low_light": low_light, "fog": fog,
    "jpeg": jpeg_compression, "color_jitter": color_jitter,
    "red_filter": red_filter, "blue_filter": blue_filter,
    "green_filter": green_filter, "warm_filter": warm_filter,
    "cool_filter": cool_filter, "combined": combined,
}


def apply(img: np.ndarray, dtype: str, severity: int = 1) -> np.ndarray:
    if dtype not in REGISTRY:
        raise KeyError(f"unknown degradation '{dtype}', valid: {list(REGISTRY)}")
    return REGISTRY[dtype](img, severity)


def random_degrade(img: np.ndarray, severity_range=(1, 4), exclude_combined=True):
    pool = AUGMENT_TYPES if exclude_combined else ALL_TYPES
    dtype = pool[np.random.randint(len(pool))]
    sev = np.random.randint(severity_range[0], severity_range[1] + 1)
    return apply(img, dtype, sev), dtype, sev
