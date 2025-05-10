from pywoe import PyWOE
import pandas

def test_pywoe(iris_data, iris_woe_p_005):
    pywoe = PyWOE(min_leave_freq=0.05, random_state=42)
    # print(iris_data.X)
    pywoe.fit(iris_data.X, iris_data.y)
    pandas.testing.assert_frame_equal(pywoe.woe_summary(), iris_woe_p_005)
