import numpy as np
import pytest
import sklearn.datasets

classification_data = sklearn.datasets.make_classification(
    n_samples=100,
    n_features=10,
    n_informative=5,
    n_redundant=5,
    n_classes=2,
    random_state=42
)


@pytest.fixture
def classification_quartile_bins():
    """Fixture providing sample quartile bins for testing."""
    x, y = classification_data
    q_bins = np.percentile(x[:, 0], q=np.arange(25, 100, 25))
    q_bins_list = [{'bin': 1,
  'freq': 25,
  'miss': 0,
  'bads': np.int64(7),
  'minx': np.float64(-4.735288259320994),
  'maxx': np.float64(-0.3067000649220044)},
 {'bin': 2,
  'freq': 25,
  'miss': 0,
  'bads': np.int64(14),
  'minx': np.float64(-0.26229478380375704),
  'maxx': np.float64(0.819932606011428)},
 {'bin': 3,
  'freq': 25,
  'miss': 0,
  'bads': np.int64(12),
  'minx': np.float64(0.8208031413681565),
  'maxx': np.float64(1.605053461769832)},
 {'bin': 4,
  'freq': 25,
  'miss': 0,
  'bads': np.int64(17),
  'minx': np.float64(1.661766509527523),
  'maxx': np.float64(5.5787090566957716)}]

    return x[:, 0], y, q_bins, q_bins_list

@pytest.fixture
def sample_cuts():
    """Fixture providing sample cut points for testing."""
    return [3, 6, 9]


@pytest.fixture
def sample_bins():
    """Fixture providing sample binning results for testing."""
    return [
        {"bin": 1, "freq": 100, "miss": 0, "bads": 20},
        {"bin": 2, "freq": 100, "miss": 0, "bads": 40},
        {"bin": 3, "freq": 100, "miss": 0, "bads": 60},
    ]
