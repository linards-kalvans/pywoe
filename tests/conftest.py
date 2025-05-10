import numpy
import pandas
import pytest
import pickle
import sklearn.datasets

n_nas = 5
rng = numpy.random.default_rng(42)

iris = sklearn.datasets.load_iris()
X = pandas.DataFrame(numpy.concatenate([iris.data, numpy.repeat(numpy.nan, 4 * n_nas).reshape(-1, iris.data.shape[1])], axis=0))
y = pandas.Series(numpy.concatenate([(iris.target == 1).astype(int), numpy.array([rng.integers(0, 2) for _ in range(n_nas)])], axis=0))

categories = ["A", "B", "C", "D", "E", "F"]
probabilities = [
    [0.22, 0.2, 0.18, 0.16, 0.14, 0.1],
    [0.12, 0.16, 0.22, 0.22, 0.16, 0.12],
    [0.1, 0.14, 0.16, 0.18, 0.2, 0.22]
]

rng_cat = numpy.random.default_rng(42)
# X = numpy.concatenate([X, numpy.array([rng_cat.choice(categories, size=1, p=probabilities[target])[0] for target in y]).reshape(-1, 1)], axis=1)
X["categorical"] = numpy.array([rng_cat.choice(categories, size=1, p=probabilities[target])[0] for target in y])
for i in range(10):
    X.loc[rng.integers(0, len(X)), "categorical"] = numpy.nan
X["static"] = 1

@pytest.fixture
def iris_data():
    class IrisData:
        X = X
        y = y
    return IrisData()

@pytest.fixture
def iris_woe_p_005():
    return pandas.read_parquet("tests/reference_data/woe_summary_iris_p_005.parquet")

@pytest.fixture
def iris_fitted_p_005():
    return pandas.read_parquet("tests/reference_data/iris_fitted_p_005.parquet")

@pytest.fixture
def iris_splits_p_005():
    with open("tests/reference_data/splits_iris_p_005.pickle", "rb") as f:
        return pickle.load(f)

@pytest.fixture
def iris_fitted_p_005_np():
    return pandas.read_parquet("tests/reference_data/iris_fitted_p_005_np.parquet")

@pytest.fixture
def iris_woe_p_0025():
    return pandas.read_parquet("tests/reference_data/woe_summary_iris_p_0025.parquet")

@pytest.fixture
def iris_fitted_p_0025():
    return pandas.read_parquet("tests/reference_data/iris_fitted_p_0025.parquet")

@pytest.fixture
def iris_woe_p_005_np_1():
    return pandas.read_parquet("tests/reference_data/woe_summary_iris_p_005_np_1.parquet")

@pytest.fixture
def iris_fitted_p_005_np_1():
    return pandas.read_parquet("tests/reference_data/iris_fitted_p_005_np_1.parquet")
