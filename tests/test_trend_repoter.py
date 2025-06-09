import numpy as np

from mathematics.stata import trend_segments


def test_trend_segments_PositiveToNegativeTransition():
    y = np.array([0, 0, 0, 0, 1, 2, 3, -1, -2, -3])
    expected_segments = [(0, 4), (4, 7), (7, 10)]
    assert trend_segments(ary=y) == expected_segments


def test_trend_segments_NegativeToPositiveTransition():
    y = np.array([0, 0, 0, 0, -1, -2, -3, 1, 2, 3])
    expected_segments = [(0, 4), (4, 7), (7, 10)]
    assert trend_segments(y) == expected_segments


def test_trend_segments_ZeroFluctuations():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    expected_segments = [(0, 10)]
    assert trend_segments(y) == expected_segments


def test_trend_segments_PositiveValues():
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected_segments = [(0, 10)]
    assert trend_segments(y) == expected_segments


def test_trend_segments_NegativeValues():
    y = np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10])
    expected_segments = [(0, 10)]
    assert trend_segments(y) == expected_segments


def test_trend_segments_EmptyArrays():
    x = np.array([])
    y = np.array([])
    expected_segments = []
    assert trend_segments(y) == expected_segments


def test_trend_segments_SingleElement():
    y = np.array([0])
    expected_segments = [(0, 1)]
    assert trend_segments(y) == expected_segments
