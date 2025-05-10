from pywoe import PyWOE
import pandas
import numpy
import pytest

def test_pywoe(iris_data, iris_woe_p_005, iris_fitted_p_005, iris_splits_p_005):
    woe = PyWOE(min_leave_freq=0.05, random_state=42)
    # print(iris_data.X)
    woe.fit(iris_data.X, iris_data.y)
    pandas.testing.assert_frame_equal(woe.woe_summary(), iris_woe_p_005, check_dtype=False)
    assert woe.splits() == iris_splits_p_005
    pandas.testing.assert_frame_equal(woe.transform(iris_data.X), iris_fitted_p_005, check_dtype=False)

def test_pywoe_numpy(iris_data, iris_woe_p_005, iris_fitted_p_005_np):
    woe = PyWOE(min_leave_freq=0.05, random_state=42)
    X = iris_data.X.drop(columns=["categorical", "static"])
    woe.fit(numpy.array(X), iris_data.y.values)
    reference = iris_woe_p_005.query("variable != 'categorical' and variable != 'static'").reset_index(drop=True)
    pandas.testing.assert_frame_equal(woe.woe_summary(), reference, check_dtype=False)
    pandas.testing.assert_frame_equal(woe.transform(numpy.array(X)), iris_fitted_p_005_np, check_dtype=False)

def test_pywoe_fit_transform(iris_data, iris_woe_p_005, iris_fitted_p_005):
    woe = PyWOE(min_leave_freq=0.05, random_state=42)
    pandas.testing.assert_frame_equal(woe.fit_transform(iris_data.X, iris_data.y), iris_fitted_p_005, check_dtype=False)
    pandas.testing.assert_frame_equal(woe.woe_summary(), iris_woe_p_005, check_dtype=False)

def test_not_fitted(iris_data):
    woe = PyWOE(min_leave_freq=0.05, random_state=42)
    with pytest.raises(ValueError):
        woe.transform(iris_data.X)

def test_missing_columns(iris_data):
    woe = PyWOE(min_leave_freq=0.05, random_state=42)
    woe.fit(iris_data.X, iris_data.y)
    X = iris_data.X.copy()
    X["new_column"] = 1
    with pytest.raises(ValueError):
        woe.transform(X)

def test_pywoe_p_0025(iris_data, iris_woe_p_0025, iris_fitted_p_0025):
    woe = PyWOE(min_leave_freq=0.025, random_state=42)
    pandas.testing.assert_frame_equal(woe.fit_transform(iris_data.X, iris_data.y), iris_fitted_p_0025, check_dtype=False)
    pandas.testing.assert_frame_equal(woe.woe_summary(), iris_woe_p_0025, check_dtype=False)

def test_new_category(iris_data):
    woe = PyWOE()
    woe.fit(iris_data.X, iris_data.y)
    X = iris_data.X.copy()
    X = pandas.concat([X, pandas.DataFrame({"categorical": ["new"]})], axis=0)
    with pytest.raises(ValueError):
        woe.transform(X)

def test_np_1(iris_data, iris_woe_p_005_np_1, iris_fitted_p_005_np_1):
    woe = PyWOE(min_leave_freq=0.05, random_state=42)
    woe.fit(numpy.array(iris_data.X["0"]), iris_data.y)
    pandas.testing.assert_frame_equal(woe.woe_summary(), iris_woe_p_005_np_1, check_dtype=False)
    pandas.testing.assert_frame_equal(woe.transform(numpy.array(iris_data.X["0"])), iris_fitted_p_005_np_1, check_dtype=False)

def test_invalid_input():
    with pytest.raises(ValueError):
        # X:3D array
        PyWOE().fit(numpy.array([1, 2, 3] * 10).reshape(-1, 1, 1), numpy.array([1, 2, 3] * 10))
    with pytest.raises(ValueError):
        # X:dict
        PyWOE().fit({}, numpy.array([1, 2, 3] * 10))
    with pytest.raises(ValueError):
        # X:1D array, y:2D array
        PyWOE().fit(numpy.array([1, 2, 3] * 10), numpy.array([1, 2, 3] * 10).reshape(-1, 1))
    with pytest.raises(ValueError):
        # X: 2D array, y: dict
        PyWOE().fit(numpy.array([1, 2, 3] * 10).reshape(-1, 1), {})
    with pytest.raises(ValueError):
        # len(X) != len(y)
        PyWOE().fit(numpy.array([1, 2, 3] * 10), numpy.array([1, 2, 3] * 9))
        
