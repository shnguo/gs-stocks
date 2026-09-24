import json

import numpy as np
import pandas as pd
import pytest

from quant_research.development import accepted_schedule
from quant_research.models import date_weights
from quant_research.storage import file_hash


def test_weight_objective_gives_each_date_same_total_contribution():
    rows = pd.DataFrame({'date': ['a', 'a', 'b', 'b', 'b']})
    weights = date_weights(rows)
    np.testing.assert_allclose(weights.mean(), 1)
    np.testing.assert_allclose(weights[:2].sum(), weights[2:].sum())


def test_development_rejects_changed_proof_or_holdout(tmp_path):
    p = tmp_path / 'schedule.json'
    p.write_text(json.dumps({'development': [{'train_start': '2015-09-01',
        'validation_start': '2020-09-01', 'test_start': '2021-09-01',
        'test_end': '2021-11-30', 'purpose': 'development'}], 'sealed_holdout_start': '2025-08-07'}))
    a = {'passed': True, 'formal_ready': False, 'universe_scope': 'all_a_shares',
         'mode': 'reconstructed_history_development_training', 'evidence': {'schedule.json': file_hash(p)}}
    assert accepted_schedule(tmp_path, a, 0).test_end == '2021-11-30'
    d = json.loads(p.read_text())
    d['sealed_holdout_start'] = '2021-10-01'
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='evidence changed'):
        accepted_schedule(tmp_path, a, 0)
    a['evidence']['schedule.json'] = file_hash(p)
    with pytest.raises(ValueError, match='holdout'):
        accepted_schedule(tmp_path, a, 0)
