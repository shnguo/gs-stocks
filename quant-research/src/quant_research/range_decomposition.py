"""Arithmetic decomposition of paired high/low point errors."""
import numpy as np


def point_components(high, low, actual_high, actual_low):
    high, low, actual_high, actual_low = np.broadcast_arrays(*[
        np.asarray(x, dtype=float) for x in [high, low, actual_high, actual_low]])
    if not all(np.isfinite(x).all() for x in [high, low, actual_high, actual_low]):
        raise ValueError('Finite endpoint prices required')
    if (high < low).any() or (actual_high < actual_low).any():
        raise ValueError('High must not be below low')
    center = (high + low) / 2
    half_width = (high - low) / 2
    center_error = center - (actual_high + actual_low) / 2
    half_width_error = half_width - (actual_high - actual_low) / 2
    return dict(center=center, half_width=half_width, center_error=center_error,
        half_width_error=half_width_error, center_mae=np.abs(center_error),
        half_width_mae=np.abs(half_width_error), center_mse=center_error**2,
        half_width_mse=half_width_error**2,
        endpoint_mae=(np.abs(high-actual_high)+np.abs(low-actual_low))/2,
        endpoint_mse=((high-actual_high)**2+(low-actual_low)**2)/2)


def attribute_mae_change(base_center_error, base_width_error, new_center_error, new_width_error):
    """Average both component replacement orders; arithmetic, not causal attribution."""
    bc, bw, nc, nw = np.broadcast_arrays(*[np.abs(np.asarray(x, float)) for x in
        [base_center_error, base_width_error, new_center_error, new_width_error]])
    before, after = np.maximum(bc, bw), np.maximum(nc, nw)
    center_first, width_first = np.maximum(nc, bw), np.maximum(bc, nw)
    return dict(center=(center_first-before+after-width_first)/2,
                half_width=(width_first-before+after-center_first)/2,
                total=after-before)
