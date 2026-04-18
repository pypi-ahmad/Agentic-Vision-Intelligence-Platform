"""Tests for frame utilities."""

import numpy as np
from PIL import Image

from src.utils.frame_utils import bgr_to_rgb, numpy_to_pil, pil_to_numpy, resize_frame, rgb_to_bgr


class TestColorConversion:
    def test_bgr_to_rgb_and_back(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # blue channel
        rgb = bgr_to_rgb(img)
        assert rgb[0, 0, 2] == 255  # now in red channel
        back = rgb_to_bgr(rgb)
        assert back[0, 0, 0] == 255


class TestPilConversion:
    def test_roundtrip(self):
        arr = np.random.randint(0, 255, (20, 30, 3), dtype=np.uint8)
        pil = numpy_to_pil(arr)
        assert isinstance(pil, Image.Image)
        back = pil_to_numpy(pil)
        assert back.shape == arr.shape


class TestResize:
    def test_no_resize_small(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out = resize_frame(img, max_dim=200)
        assert out.shape == (100, 100, 3)

    def test_resize_large(self):
        img = np.zeros((2000, 3000, 3), dtype=np.uint8)
        out = resize_frame(img, max_dim=1280)
        assert max(out.shape[:2]) <= 1280

    def test_aspect_ratio_preserved(self):
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        out = resize_frame(img, max_dim=1000)
        h, w = out.shape[:2]
        ratio_in = 1000 / 2000
        ratio_out = h / w
        assert abs(ratio_in - ratio_out) < 0.02
