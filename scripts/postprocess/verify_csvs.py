import sys
import pandas as pd
import re

def no_duplicate_rows(cases_df, dims_df):
    assert cases_df.duplicated().sum() == 0, "Duplicate rows found in cases_df"
    assert dims_df.duplicated().sum() == 0, "Duplicate rows found in dims_df"
    print("No duplicate rows found")

def column_sums_match(cases_df, dims_df):
    var_pattern = re.compile(r'^(.*)_Var\d+$')
    columns_to_sum = {}

    for col in cases_df.columns:
        match = var_pattern.match(col)
        if match:
            columns_to_sum.setdefault(match.group(1), []).append(col)
        elif col in dims_df.columns:
            pd.testing.assert_series_equal(
                cases_df[col],
                dims_df[col],
                check_names=False,
                rtol=1e-3,   # relative tolerance
                atol=0.1     # absolute tolerance of 0.1
            )

    for prefix, cols in columns_to_sum.items():
        if prefix in dims_df.columns:
            pd.testing.assert_series_equal(
                cases_df[cols].sum(axis=1),
                dims_df[prefix],
                check_names=False,
                rtol=1e-3,
                atol=0.1      # allow up to ±0.1 absolute difference
            )
        else:
            raise ValueError(f"Column {prefix} not found in dims_df")
        print("Column sums match")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python verify_csvs.py <cases_csv> <dims_csv>")
        sys.exit(1)

    cases_df = pd.read_csv(sys.argv[1])
    dims_df = pd.read_csv(sys.argv[2])

    no_duplicate_rows(cases_df, dims_df)
    column_sums_match(cases_df, dims_df)
