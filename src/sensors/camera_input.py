"""Camera subscription helpers: raw sensor_msgs/Image or CompressedImage."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image

CameraMsg = Union[Image, CompressedImage]


def resolve_camera_transport(explicit: str, topic: str) -> str:
    """Resolve transport to ``raw`` or ``compressed``.

    ``explicit`` may be ``auto``, ``raw``, or ``compressed``. In ``auto`` mode the
    topic suffix ``/compressed`` selects compressed; everything else is raw.
    """
    mode = (explicit or "auto").strip().lower()
    if mode in ("compressed", "raw"):
        return mode
    if mode == "auto":
        normalized = topic.rstrip("/")
        if normalized.endswith("/compressed") or normalized.endswith("compressed"):
            return "compressed"
        return "raw"
    raise ValueError(
        f"camera_transport must be auto, raw, or compressed (got {explicit!r})"
    )


def camera_message_type(transport: str):
    if transport == "compressed":
        return CompressedImage
    if transport == "raw":
        return Image
    raise ValueError(f"unknown camera transport {transport!r}")


def parse_camera_msg(
    msg: CameraMsg, bridge: CvBridge
) -> Tuple[Optional[Image], Optional[np.ndarray]]:
    """Decode camera input to an rgb8 ``Image`` and optional OpenCV array."""
    if isinstance(msg, CompressedImage):
        return _parse_compressed(msg, bridge)
    return _parse_raw(msg, bridge)


def _parse_compressed(
    msg: CompressedImage, bridge: CvBridge
) -> Tuple[Optional[Image], Optional[np.ndarray]]:
    try:
        cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="rgb8")
    except Exception:
        try:
            cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        except Exception:
            return None, None

    if cv_img is None or (hasattr(cv_img, "size") and cv_img.size == 0):
        return None, None

    try:
        img_msg = bridge.cv2_to_imgmsg(cv_img, encoding="rgb8")
    except Exception:
        return None, None

    img_msg.header = msg.header
    return img_msg, cv_img


def _parse_raw(
    msg: Image, bridge: CvBridge
) -> Tuple[Optional[Image], Optional[np.ndarray]]:
    try:
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    except Exception:
        return None, None

    if img is None:
        return None, None

    if hasattr(img, "size") and img.size == 0:
        return None, None

    if "bgr" in msg.encoding:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif "rgba" in msg.encoding:
        if img.ndim == 3 and img.shape[2] >= 3:
            img = img[..., :3]
        else:
            return None, None
    elif "rgb" not in msg.encoding:
        # mono / yuv / etc. — let cv_bridge try rgb8 for downstream CNN
        try:
            img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception:
            return None, None

    try:
        img_msg = bridge.cv2_to_imgmsg(img, encoding="rgb8")
    except Exception:
        return None, None

    img_msg.header = msg.header
    return img_msg, img
