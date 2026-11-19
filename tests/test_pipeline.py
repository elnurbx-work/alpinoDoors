import numpy as np

from alpino import PRESETS, get_door, place_door
from alpino.geometry import mask_to_quad, order_corners


def test_order_corners():
    q = order_corners([[10, 90], [90, 10], [10, 10], [90, 90]])
    assert q.tolist() == [[10, 10], [90, 10], [90, 90], [10, 90]]


def test_mask_to_quad():
    m = np.zeros((200, 200), np.uint8)
    m[40:180, 50:120] = 1
    q = mask_to_quad(m)
    assert np.allclose(q, [[50, 40], [119, 40], [119, 179], [50, 179]], atol=2)


def test_all_presets_render_and_place():
    photo = np.full((600, 400, 3), 200, np.uint8)
    quad = [[100, 100], [300, 110], [295, 560], [105, 570]]
    for k in PRESETS:
        door = get_door(k, 1588, 2518)
        assert door.shape[2] == 4 and door[..., 3].max() == 255
        out = place_door(photo, quad, door, 1588, 2518, annotate=True)
        assert out.shape == photo.shape
        assert out[300, 50].tolist() == [200, 200, 200] or abs(int(out[300, 50, 0]) - 200) < 10
        assert out[108:112, 190:210].mean() < 150, k  # üst profil qara görünməlidir
