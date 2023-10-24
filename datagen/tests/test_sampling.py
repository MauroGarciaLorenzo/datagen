
from unittest import TestCase

from datagen import *
from datagen.tests.utils import gen_df_for_dims, linear_function, \
    dim0_func, parab_func
from datagen.src.sampling import sensitivity


class Test(TestCase):

    def setUp(self):
        """
        This code executes every time a subtest of this class is run.
        """
        variables = [(0, 10), (0, 15), (0, 20), (0, 25)]

        self.dim1 = Dimension(variables, n_cases=3, divs=1, borders=(0, 70),
                              label="Dim1")
        self.dim2 = Dimension(variables, n_cases=3, divs=2, borders=(0, 70),
                              label="Dim2")
        self.dims = [self.dim1, self.dim2]

        self.children_grid = gen_grid([self.dim1, self.dim2])

        data = {
            "Dim1": [10, 35, 70],
            "Dim2": [15, 40, 80]
        }
        self.dims_df = pd.DataFrame(data)

        self.cases_heritage_df = pd.DataFrame({
            "Dim1_Var0": [5, 15, 35],
            "Dim1_Var1": [2, 8, 20],
            "Dim1_Var2": [2, 6, 10],
            "Dim1_Var3": [1, 6, 5],
            "Dim2_Var0": [5, 10, 40],
            "Dim2_Var1": [5, 10, 20],
            "Dim2_Var2": [3, 10, 15],
            "Dim2_Var3": [2, 10, 5]
        })

    def test_generate_columns(self):
        cols = generate_columns(self.dim1)
        self.assertTrue(cols,
                        ["Dim1_Var0", "Dim1_Var1", "Dim1_Var2", "Dim1_Var3"])

    def test_gen_samples(self):
        n_samples = 100
        dimensions = self.dims
        df_samples = gen_samples(n_samples, dimensions)

        for dim in dimensions:
            self.assertTrue(all(df_samples[dim.label] >= dim.borders[0]))
            self.assertTrue(all(df_samples[dim.label] <= dim.borders[1]))

    def test_gen_grid(self):
        grid = gen_grid(self.dims)

        expected_cells = 1 * 2
        self.assertEqual(len(grid), expected_cells)

        for i, cell in enumerate(grid):
            for dim, expected_dim in zip(cell.dimensions, self.dims):
                expected_upper = expected_dim.borders[
                                     1] / expected_dim.divs * (i + 1)
                expected_lower = expected_dim.borders[
                                     1] / expected_dim.divs * i

                self.assertEqual(dim.borders[0], expected_lower)
                self.assertEqual(dim.borders[1], expected_upper)

    def test_calculate_entropy(self):
        freqs = [0.25, 0.75]
        expected_entropy = -0.25 * np.log(0.25) - 0.75 * np.log(0.75)
        result_entropy = calculate_entropy(freqs)
        self.assertAlmostEqual(result_entropy, expected_entropy, places=5)

    def test_eval_entropy(self):
        stabilities = [1, 0, 0, 0]
        entropy_parent = None
        expected_freqs = [0.25, 0.75]
        expected_entropy = calculate_entropy(expected_freqs)
        expected_delta_entropy = 1

        result_entropy, result_delta_entropy = eval_entropy(stabilities,
                                                            entropy_parent)

        self.assertAlmostEqual(result_entropy, expected_entropy, places=5)
        self.assertEqual(result_delta_entropy, expected_delta_entropy)

    def test_assign_cases(self):
        total_cases_df, _ = get_children_parameters(self.children_grid,
                                                    self.dims_df,
                                                    self.cases_heritage_df)

        # (Dim1: 0-70, Dim2: 0-35)
        self.assertEqual(len(total_cases_df[0]), 1)
        pd.testing.assert_frame_equal(total_cases_df[0],
                                      self.cases_heritage_df.iloc[:1])

        # (Dim1: 0-70, Dim2: 35-70)
        self.assertEqual(len(total_cases_df[1]), 2)
        pd.testing.assert_frame_equal(total_cases_df[1],
                                      self.cases_heritage_df.iloc[1:3])

    def test_sensitivity(self):
        variables = [(0, 10), (0, 10), (0, 10), (0, 10)]
        dim1 = Dimension(variables, n_cases=3, divs=1, borders=(0, 70),
                         label="Dim1")
        dim2 = Dimension(variables, n_cases=3, divs=2, borders=(0, 70),
                         label="Dim2")
        dim3 = Dimension(variables, n_cases=3, divs=2, borders=(0, 70),
                         label="Dim3")
        dims = [dim1, dim2, dim3]
        cases_df = gen_df_for_dims(dims, 1000)

        linear_cases_df = cases_df.copy()
        parab_cases_df = cases_df.copy()
        dim0_cases_df = cases_df.copy()

        linear_cases_df["Stability"] = (
            linear_cases_df.apply(linear_function, axis=1))
        parab_cases_df["Stability"] = parab_cases_df.apply(parab_func, axis=1)
        dim0_cases_df["Stability"] = dim0_cases_df.apply(dim0_func, axis=1)

        dims_linear = sensitivity(linear_cases_df, dims)
        dims_linear_divs = [dim.divs for dim in dims_linear]
        dims_parab = sensitivity(parab_cases_df, dims)
        dims_parab_divs = [dim.divs for dim in dims_parab]
        dims_dim0 = sensitivity(dim0_cases_df, dims)
        dims_dim0_divs = [dim.divs for dim in dims_dim0]

        self.assertEqual(dims_linear_divs, [1, 1, 2])
        self.assertEqual(dims_parab_divs, [1, 2, 1])
        self.assertEqual(dims_dim0_divs, [2, 1, 1])

