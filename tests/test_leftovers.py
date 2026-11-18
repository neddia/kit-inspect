"""The OpenCV unknown-object check on drawn images. No camera, no model.

Run: uv run pytest tests/test_leftovers.py
"""

import cv2
import numpy as np

from src.inspection.leftovers import DEFAULTS, find_leftovers

FLOOR = (640, 360, 300)   # x, y, radius of the flat floor


def floor_with(draw):
    img = np.full((720, 1280, 3), 200, np.uint8)   # plain grey-white floor, no colour
    draw(img)
    return img


def silver_coin(img):
    cv2.circle(img, (500, 360), 40, (230, 230, 230), -1)   # a bit brighter than the floor: not darker, not coloured


def test_empty_floor_has_nothing():
    assert find_leftovers(floor_with(lambda img: None), FLOOR, [], **DEFAULTS) == []


def test_silver_coin_needs_the_edge_test():
    img = floor_with(silver_coin)
    assert find_leftovers(img, FLOOR, [], **(DEFAULTS | {"edge_low": 0})) == []   # darker/coloured tests: blind
    found = find_leftovers(img, FLOOR, [], **DEFAULTS)                          # edge test: sees the outline
    assert len(found) == 1 and found[0]["box"][0] < 470 and found[0]["box"][2] > 530


def test_anything_inside_a_part_box_is_ignored():
    assert find_leftovers(floor_with(silver_coin), FLOOR, [[440, 300, 560, 420]], **DEFAULTS) == []


def test_dark_blob_is_found_without_edges():
    img = floor_with(lambda im: cv2.circle(im, (700, 400), 15, (60, 60, 60), -1))
    assert len(find_leftovers(img, FLOOR, [], **(DEFAULTS | {"edge_low": 0}))) == 1
