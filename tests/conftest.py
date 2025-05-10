import numpy
import pandas
import pytest
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

@pytest.fixture
def iris_data():
    class IrisData:
        X = X
        y = y
    return IrisData()

@pytest.fixture
def iris_woe_p_005():
    return pandas.DataFrame({
        "variable": ["0",  "0",  "0",  "0",  "1",  "2",  "2",  "2",  "3",  "3",  "categorical",  "categorical",  "categorical",  "categorical",  "categorical", "categorical"],
        "freq": [16, 25, 11, 103, 155, 48, 8, 99, 48, 107, 29, 25, 38, 29, 22, 12],
        "events": [0.5,  4.5,  2.5,  46.5,  52.5,  0.5,  3.5,  49.5,  0.5,  52.5,  8.5,  8.5,  13.5,  14.5,  7.5,  2.5],
        "nas": [0.0,  0.0,  0.0,  5.0,  5.0,  0.0,  5.0,  0.0,  0.0,  5.0,  numpy.nan,  numpy.nan,  numpy.nan,  numpy.nan,  numpy.nan,  numpy.nan],
        "non_events": [16.5,  21.5,  9.5,  57.5,  103.5,  48.5,  5.5,  50.5,  48.5,  55.5,  21.5,  17.5,  25.5,  15.5,  15.5,  10.5],
        "eventrate": [0.03125,  0.18,  0.22727272727272727,  0.45145631067961167,  0.3387096774193548,  0.010416666666666666,  0.4375,  0.5,  0.010416666666666666,  0.49065420560747663,  0.29310344827586204,  0.34,  0.35526315789473684,  0.5,  0.3409090909090909,  0.20833333333333334],
        "woe": [-2.8315312578732312,  -0.898999234764094,  -0.6700247631390911,  0.45264366838325476,  0.0,  -3.905205561000478,  0.2175202937598476,  0.6495047507962353,  -3.900611992914132,  0.61852913443444,  -0.2718808627577502,  -0.06602880855360137,  0.020117142159599477,  0.589414534380924,  -0.06983109450334,  -0.7789786164097264],
        "iv": [0.41873702993416306,  0.10916419279278285,  0.029601623133393706,  0.14190019762808384,  0.0,  1.775966576728295,  0.0027818630780895408,  0.2870681463520095,  1.78223717419344,  0.2826135023717593,  0.01312769517432362,  0.0006965303132155201,  9.834280472531469e-05,  0.06920313015158532,  0.0006887286336092714,  0.04175485722607882],
        "group": ["[-inf, 4.9)",  "[4.9, 5.2)",  "[5.2, 5.5)",  "[5.5, inf) OR NA",  "[-inf, inf) OR NA",  "[-inf, 1.9)",  "[1.9, 3.3) OR NA",  "[3.3, inf)",  "[-inf, 0.5)",  "[0.5, inf) OR NA",  "A",  "B",  "C",  "D",  "E",  "F"]
    })
