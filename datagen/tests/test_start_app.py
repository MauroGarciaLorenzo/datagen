from unittest import TestCase

import pandas as pd

from datagen import Dimension, start, dummy


class Test(TestCase):
    def setUp(self):
        p_sg = [(0, 0.1), (0, 0.1), (0, 0.1), (10, 25)]
        p_cig = [(0, 1), (0, 1.5), (0, 1.5), (0, 2)]
        tau_f_g_for = [(0., 2)]
        tau_v_g_for = [(0., 2)]
        tau_p_g_for = [(0., 2)]
        tau_q_g_for = [(0., 2)]
        self.n_samples = 3
        self.n_cases = 3
        # ax = plt.figure().add_subplot(projection='3d')
        self.ax = None
        self.use_sensitivity = False
        self.rel_tolerance = 0.01
        self.max_depth = 5
        self.func = dummy
        self.dimensions = {
            "p_sg":
                Dimension(variables=p_sg, n_cases=self.n_cases, divs=2,
                      borders=(10, 25.3), is_true_dimension=True),
            "p_cig":
                Dimension(variables=p_cig, n_cases=self.n_cases, divs=1,
                          borders=(0, 6), is_true_dimension=True),
            "tau_f_g_for":
                Dimension(variables=tau_f_g_for, n_cases=self.n_cases, divs=1,
                          borders=(0, 2), is_true_dimension=True),
            "tau_v_g_for":
                Dimension(variables=tau_v_g_for, n_cases=self.n_cases, divs=1,
                          borders=(0, 2), is_true_dimension=True),
            "tau_p_g_for":
                Dimension(variables=tau_p_g_for, n_cases=self.n_cases, divs=1,
                          borders=(0, 2), is_true_dimension=True),
            "tau_q_g_for":
                Dimension(variables=tau_q_g_for, n_cases=self.n_cases, divs=1,
                          borders=(0, 2), is_true_dimension=True)
        }

    """
                
                
                """

    def test_start(self):
        cases_df, dims_df, execution_logs = \
            start(self.dimensions, self.n_samples, self.rel_tolerance,
                  self.func, self.max_depth,
                  use_sensitivity=self.use_sensitivity, ax=self.ax)
        # assert that cases_df must have 21 columns and dims_df 8
        self.assertEqual(cases_df.shape[1], 21)
        self.assertEqual(dims_df.shape[1], 8)

        # assert that the sum of the variables of a concrete dimension in
        # cases_df almost equals the value in dims_df
        dim_labels = list(set(col.rsplit('_Var')[0]
                          for col in cases_df.columns if '_Var' in col))
        dims_expected_df = pd.DataFrame()
        for label in dim_labels:
            matching_columns = (
                cases_df.filter(
                    regex=r'^' + label + r'_*', axis=1).sum(axis=1))
            dims_expected_df[label] = matching_columns
        dims_expected_df.columns = dim_labels
        dims_expected_df = dims_expected_df[dims_df.columns]
        pd.testing.assert_frame_equal(dims_expected_df, dims_df,
                                      check_exact=False)

        # assert that each value is within dimension borders
        for idx, row in dims_df.iterrows():
            for label, value in row.items():
                if label == "g_for" or label == "g_fol":
                    dim = next(
                        (d for l, d in self.dimensions.items()
                         if l == "p_cig"), None)
                else:
                    dim = next((d for l,d in self.dimensions.items()
                                if l == label), None)
                if not dim.borders[0] <= value <= dim.borders[1]:
                    pass
                self.assertTrue(dim.borders[0] <= value <= dim.borders[1])

        # assert that there is no null value in cases_df and that every value
        # is within associated variable limits
        for idx, row in cases_df.iterrows():
            for label, value in row.items():
                if label != "Stability":
                    self.assertIsNotNone(value)
                    dim_label = label.rsplit('_Var')[0]
                    var_idx = int(label.rsplit('_Var')[1])
                    if dim_label == "g_for" or dim_label == "g_fol":
                        dim = next(
                            (d for l,d in self.dimensions.items()
                             if l == "p_cig"), None)
                    else:
                        dim = next((d for l,d in self.dimensions.items()
                                    if l == dim_label), None)

                    self.assertTrue(dim.variables[var_idx][0] <= value <=
                                    dim.variables[var_idx][1])
