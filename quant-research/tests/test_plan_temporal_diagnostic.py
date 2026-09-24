import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from plan_temporal_diagnostic import offset_effect


def test_offset_mse_decomposition_keeps_unknowns_out():
    y = np.array([.01, -.03, np.nan])
    raw = np.array([.02, -.02, 100.])
    helpful = offset_effect(y, raw, raw-.01, -.01)
    assert helpful['corrected_mse'] == pytest.approx(0)
    assert helpful['mse_change'] == pytest.approx(-.0001)
    harmful = offset_effect(y, raw, raw+.02, .02)
    assert harmful['mse_change'] == pytest.approx(.0008)
    assert harmful['offset_cross_term']+harmful['offset_square_term'] == pytest.approx(.0008)
    assert harmful['actual_mean'] == pytest.approx(-.01)


def test_offset_rejects_changed_correction_and_preserves_empty():
    x = np.array([.02])
    with pytest.raises(AssertionError):
        offset_effect(x, x, x+.02, .01)
    values = offset_effect(np.array([np.nan]), x, x, 0)
    assert all(np.isnan(v) for v in values.values())
