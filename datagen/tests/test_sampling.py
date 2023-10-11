
from unittest import TestCase

from datagen import generate_columns, Dimension


class Test(TestCase):

    def test_generate_columns(self):
        dim = Dimension()
        cols = generate_columns(dim)
        self.assertTrue(len(cols), ["Dim1", "Dim2", "Dim3"])
