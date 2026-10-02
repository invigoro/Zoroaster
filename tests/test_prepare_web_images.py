import unittest

from PIL import Image

from scripts.prepare_web_images import flatten, key_out_white, light_mask


class ImagesTest(unittest.TestCase):
    def test_white_is_keyed_out_and_black_kept(self):
        art = Image.new("RGB", (3, 1))
        art.putdata([(255, 255, 255), (0, 0, 0), (128, 128, 128)])
        keyed = key_out_white(art)
        self.assertEqual(keyed.mode, "RGBA")
        self.assertEqual(list(keyed.getdata()), [(0, 0, 0, 0), (0, 0, 0, 255), (0, 0, 0, 127)])

    def test_light_parts_kept_and_black_lines_and_background_cleared(self):
        art = Image.new("RGBA", (5, 1))
        art.putdata([(255, 255, 255, 255), (0, 0, 0, 255), (0, 0, 0, 0), (128, 128, 128, 255), (255, 255, 255, 128)])
        self.assertEqual([a for *_, a in light_mask(art).getdata()], [255, 0, 0, 128, 128])
        self.assertEqual({p[:3] for p in light_mask(art).getdata()}, {(255, 255, 255)})  # white ink throughout

    def test_flatten_onto_black(self):
        art = Image.new("RGBA", (2, 1))
        art.putdata([(255, 255, 255, 128), (10, 20, 30, 0)])
        self.assertEqual(list(flatten(art).getdata()), [(128, 128, 128), (0, 0, 0)])


if __name__ == "__main__":
    unittest.main()
