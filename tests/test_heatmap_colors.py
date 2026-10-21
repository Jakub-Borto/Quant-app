"""The Explore heatmap's coloring (heatmap_model.color_positions): a split
value gets the scale's middle color, values above it use the upper half
(bent by the "above" curve), values below it the lower half (bent by the
"below" curve)."""

import numpy as np

from modules.optimizer.backend.heatmap_model import (HEATMAP_COLORSCALE, METRIC_PIVOTS,
                                                     color_positions,
                                                     two_curve_colorscale)

Z = np.linspace(-300.0, 100.0, 41)          # below the split: 3x longer than above


def test_split_in_the_middle_with_flat_curves_is_the_linear_scale():
    lin = (Z - Z.min()) / (Z.max() - Z.min())
    np.testing.assert_allclose(color_positions(Z, -300, 100, -100.0), lin, atol=1e-12)
    np.testing.assert_allclose(color_positions(Z, -300, 100, None), lin, atol=1e-12)
    assert two_curve_colorscale(-300, 100, None) is HEATMAP_COLORSCALE
    assert two_curve_colorscale(-300, 100, 0.0) is not HEATMAP_COLORSCALE


def test_the_split_gets_the_middle_color_and_each_side_its_half():
    pos = color_positions(Z, -300, 100, 0.0)
    assert color_positions(0.0, -300, 100, 0.0) == 0.5          # the split: yellow
    assert pos[0] == 0.0 and pos[-1] == 1.0                     # the extremes
    assert (pos[Z > 0] > 0.5).all() and (pos[Z < 0] < 0.5).all()
    np.testing.assert_allclose(color_positions(50.0, -300, 100, 0.0), 0.75)   # halfway up
    np.testing.assert_allclose(color_positions(-150.0, -300, 100, 0.0), 0.25)  # halfway down


def test_each_curve_only_bends_its_own_side():
    base = color_positions(Z, -300, 100, 0.0)
    good = color_positions(Z, -300, 100, 0.0, good=2.0)
    bad = color_positions(Z, -300, 100, 0.0, bad=-1.5)
    above, below = Z > 0, Z < 0
    np.testing.assert_allclose(good[below], base[below])
    assert (good[above][:-1] < base[above][:-1]).all()         # exp-like: red only at the top
    assert good[-1] == 1.0
    np.testing.assert_allclose(bad[above], base[above])
    assert (bad[below][1:] < base[below][1:]).all()            # log-like: blue reaches the split
    assert bad[0] == 0.0
    for curve in (good, bad):
        assert (np.diff(curve) >= 0).all()                      # still ordered


def test_split_outside_the_range_uses_one_half():
    allgood = np.linspace(5.0, 50.0, 10)                         # every cell above the split
    pos = color_positions(allgood, 5, 50, 0.0)
    assert (pos >= 0.5).all() and pos[-1] == 1.0
    np.testing.assert_allclose(pos, 0.5 + 0.5 * allgood / 50)
    assert METRIC_PIVOTS["profit_factor"] == 1.0 and METRIC_PIVOTS["total_trades"] is None
    stops = two_curve_colorscale(-300, 100, 0.0, good=1.0, bad=-1.0)
    assert stops[0][0] == 0.0 and stops[-1][0] == 1.0 and len(stops) == 129
