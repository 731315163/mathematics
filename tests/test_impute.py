import numpy as np
# import pandas as pd
# import pytest

from mathematics.impute import forward_fill  # 确保这个导入与你的项目结构相匹配

def test_middle_nans_filled():
    """Test filling NaN values between valid numbers"""
    input_array = np.array([1.0, np.nan, np.nan, 4.0, np.nan, 6.0])
    expected = np.array([1.0, 1.0, 1.0, 4.0, 4.0, 6.0])
    result = forward_fill(input_array)
    np.testing.assert_array_equal(result, expected)

def test_fill_first_true():
    """Test filling leading NaN values when fill_first=True"""
    input_array = np.array([np.nan, np.nan, 3.0, np.nan, 5.0])
    expected = np.array([3.0, 3.0, 3.0, 3.0, 5.0])
    result = forward_fill(input_array, fill_first=True)
    np.testing.assert_array_equal(result, expected)

def test_fill_first_false():
    """Test preserving leading NaN values when fill_first=False"""
    input_array = np.array([np.nan, np.nan, 3.0, np.nan, 5.0])
    expected = np.array([np.nan, np.nan, 3.0, 3.0, 5.0])
    result = forward_fill(input_array)
    np.testing.assert_array_equal(result, expected)

def test_all_nans():
    """Test array with all NaN values"""
    input_array = np.array([np.nan, np.nan, np.nan])
    result = forward_fill(input_array)
    np.testing.assert_array_equal(result, input_array)

def test_empty_array():
    """Test empty array input"""
    input_array = np.array([])
    result = forward_fill(input_array)
    np.testing.assert_array_equal(result, input_array)

def test_no_nans():
    """Test array with no NaN values"""
    input_array = np.array([1.0, 2.0, 3.0])
    result = forward_fill(input_array)
    np.testing.assert_array_equal(result, input_array)

def test_single_nan():
    """Test array with single NaN value"""
    input_array = np.array([1.0, np.nan, 3.0])
    expected = np.array([1.0, 1.0, 3.0])
    result = forward_fill(input_array)
    np.testing.assert_array_equal(result, expected)

def test_different_data_types():
    """Test with different data types"""
    input_array = np.array([1, np.nan, 3], dtype=float)
    expected = np.array([1, 1, 3], dtype=float)
    result = forward_fill(input_array)
    np.testing.assert_array_equal(result, expected)
# ... existing code ...
# @pytest.fixture
# def df_with_na():
#     return pd.DataFrame(
#         {
#             "A": [1, 2, np.nan, 4, 5, 9],
#             "B": [5, np.nan, np.nan, 6, np.nan, 3],
#             "C": [9, 10, 11, 5, 1, np.nan],
#         }
#     )


# @pytest.fixture
# def df_without_na():
#     return pd.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8], "C": [9, 10, 11, 12]})


# def test_akimaimputedf_with_na(df_with_na: pd.DataFrame):
#     imputed_df = akima_impute_df(df_with_na)
#     print(imputed_df.head())
#     for col in ["A", "B"]:
#         assert not imputed_df[col].isna().any()


# # def test_akimaimputedf_without_na(df_without_na):
# #     imputed_df = akima_impute_df(df_without_na)
# #     pd.testing.assert_frame_equal(imputed_df, df_without_na)


# def test_akimaimputedf_specific_columns(df_with_na):
#     columns_to_impute = ["A", "C"]
#     imputed_df = akima_impute_df(df_with_na, columns=columns_to_impute)
#     assert not imputed_df["A"].isna().any()
#     assert np.isnan(imputed_df["B"][1])


# def test_akimaimputedf_method(df_with_na):
#     imputed_df_akima = akima_impute_df(df_with_na, method="akima")
#     imputed_df_makima = akima_impute_df(df_with_na, method="makima")
#     assert imputed_df_akima["B"][2] != imputed_df_makima["B"][2]
