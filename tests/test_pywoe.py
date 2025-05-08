import numpy as np
import pytest
from pywoe.pywoe import manual_bin, gbm_bin, miss_bin, add_miss, gen_rule, gen_woe

def test_manual_bin(classification_quartile_bins):
    # Test data
    x, y, q_bins, q_bins_list = classification_quartile_bins

    result = manual_bin(x, y, q_bins)

    # Check if result is a list
    assert isinstance(result, list)
    assert len(result) == len(q_bins_list)
    # Check if each bin has required keys
    required_keys = {"bin", "freq", "miss", "bads", "minx", "maxx"}
    for bin_result, q_bin in zip(result, q_bins_list):
        assert set(bin_result.keys()) == required_keys
        assert bin_result["bin"] == q_bin["bin"]
        assert bin_result["freq"] == q_bin["freq"]
        assert bin_result["miss"] == q_bin["miss"]
        assert bin_result["bads"] == q_bin["bads"]
        assert bin_result["minx"] == q_bin["minx"]
        assert bin_result["maxx"] == q_bin["maxx"]

    # Check if bins are properly ordered
    assert all(result[i]["bin"] < result[i + 1]["bin"] for i in range(len(result) - 1))


# def test_miss_bin():
#     # Test data
#     y = [0, 1, 0, 1, 1]

#     result = miss_bin(y)

#     # Check if result is a dictionary
#     assert isinstance(result, dict)

#     # Check if all required keys are present
#     required_keys = {"bin", "freq", "miss", "bads", "minx", "maxx"}
#     assert set(result.keys()) == required_keys

#     # Check values
#     assert result["freq"] == 5
#     assert result["miss"] == 5
#     assert result["bads"] == 3
#     assert np.isnan(result["minx"])
#     assert np.isnan(result["maxx"])


# def test_gen_woe():
#     # Test data
#     x = [
#         {"bin": 1, "freq": 100, "miss": 0, "bads": 20},
#         {"bin": 2, "freq": 100, "miss": 0, "bads": 40},
#         {"bin": 3, "freq": 100, "miss": 0, "bads": 60},
#     ]

#     result = gen_woe(x)

#     # Check if result is a list
#     assert isinstance(result, list)

#     # Check if each bin has required keys
#     required_keys = {"bin", "freq", "miss", "bads", "rate", "woe", "iv", "ks"}
#     for bin_result in result:
#         assert set(bin_result.keys()) == required_keys

#     # Check if WOE and IV are calculated
#     for bin_result in result:
#         assert "woe" in bin_result
#         assert "iv" in bin_result
#         assert "ks" in bin_result


# def test_gbm_bin():
#     # Test data
#     x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     y = [0, 0, 0, 1, 1, 1, 0, 0, 1, 1]

#     result = gbm_bin(x, y)

#     # Check if result is a dictionary
#     assert isinstance(result, dict)

#     # Check if result has required keys
#     assert set(result.keys()) == {"cut", "tbl"}

#     # Check if cut points are valid
#     assert isinstance(result["cut"], list)
#     assert all(isinstance(cut, (int, float)) for cut in result["cut"])

#     # Check if table has required structure
#     assert isinstance(result["tbl"], list)
#     if result["tbl"]:
#         required_keys = {
#             "bin",
#             "freq",
#             "miss",
#             "bads",
#             "rate",
#             "woe",
#             "iv",
#             "ks",
#             "rule",
#         }
#         assert set(result["tbl"][0].keys()) == required_keys
