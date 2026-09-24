import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import plan_recent_refit as refit

from quant_research.storage import write_json


def spec():
    days = [f'2024-{month:02d}-{day:02d}' for month in range(1, 6) for day in range(1, 21)]
    return dict(train=days[:96], selection=days[96:], selection_label_end='2024-05-27', asof_signal_close='2024-05-27')


def test_refit_replaces_oldest_four_without_growing_sample_dates():
    source = spec()
    actual = refit.plan_dates(source)
    assert len(actual) == len(set(actual)) == 96
    assert actual == source['train'][4:]+source['selection']
    assert set(source['train'][:4]).isdisjoint(actual)


def test_immature_selection_cannot_be_reused():
    source = spec()
    source['selection_label_end'] = '2024-05-28'
    with pytest.raises(ValueError, match='not mature'):
        refit.plan_dates(source)


def test_cached_future_labels_rejected_before_loading_values(tmp_path, monkeypatch):
    directory = tmp_path/'day-cache/2024-05-20'
    directory.mkdir(parents=True)
    write_json(directory/'input-manifest.json', dict(date='2024-05-20', first_requested_asof='2024-05-20'))
    write_json(directory/'label-manifest.json', dict(date='2024-05-20', label_end='2024-05-27', first_requested_asof='2024-05-27'))
    def forbidden(*args):
        raise AssertionError('Target array was opened')
    monkeypatch.setattr(refit, 'load', forbidden)
    with pytest.raises(ValueError, match='Immature labels'):
        refit.cached(tmp_path, {}, '2024-05-20', '2024-05-24', True)
