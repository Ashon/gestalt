import unittest

from gestalt.engine.som import utils


class TestSomUtil(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(utils.clamp(2, 0, 3), 2)
        self.assertEqual(utils.clamp(-30, 0, 3), 0)
        self.assertEqual(utils.clamp(30, 0, 3), 3)
        self.assertEqual(utils.clamp(0, 0, 3), 0)
