import numpy as np
import pandas as pd
import pytest

from quant_research.category_context import (
    encode_categories,
    fit_dictionary,
    identity_categories,
    raw_categories,
)


def test_training_only_dictionary_unknown_and_column_contract():
    train = pd.DataFrame({'stock_identity': ['b', 'a', None]})
    dictionary = fit_dictionary(train)
    assert dictionary == {'stock_identity': ['a', 'b']}
    encoded = encode_categories(pd.DataFrame({'stock_identity': ['b', 'new', None]}), dictionary)
    np.testing.assert_array_equal(encoded[:, 0], [1, -1, -1])
    with pytest.raises(ValueError, match='schema'):
        encode_categories(pd.DataFrame({'wrong': ['a']}), dictionary)


def test_identity_family_does_not_backfill_current_main_board():
    assert identity_categories('cn.xshe.002001') == ('xshe', 'sz_other_a')
    assert identity_categories('cn.xshe.000001') == ('xshe', 'sz_other_a')
    assert identity_categories('cn.xshe.301001') == ('xshe', 'sz_30')
    assert identity_categories('cn.xbse.920001') == ('xbse', 'bj_identity')
    with pytest.raises(ValueError):
        identity_categories('bad')


def test_past_snapshot_and_missing_bse_retained():
    rows = pd.DataFrame({'date': ['2024-01-04']*2,
                         'instrument_id': ['cn.xshe.000001', 'cn.xbse.920001']})
    frame = pd.DataFrame({'instrument_id': ['cn.xshe.000001'], 'industry': ['bank'],
                          'industryClassification': ['taxonomy'], 'updateDate': ['2024-01-01'],
                          'requested_date': ['2024-01-02']})
    result = raw_categories(rows, {'2024-01-04': frame})
    assert len(result) == 2 and result.industry_category.isna().sum() == 1
    for requested in ['2024-01-05', '2023-12-20']:
        wrong = frame.assign(requested_date=requested)
        with pytest.raises(ValueError, match='source'):
            raw_categories(rows, {'2024-01-04': wrong})


def test_lgbm_native_category_is_saved_and_unseen_predictable(tmp_path):
    import lightgbm as lgb
    x = np.column_stack([np.zeros(300), np.tile(np.arange(3), 100)])
    y = np.tile([0., 1., 0.], 100)
    model = lgb.LGBMRegressor(n_estimators=4, min_child_samples=5, verbosity=-1,
                             n_jobs=1, min_data_per_group=5)
    model.fit(x, y, categorical_feature=[1])
    dest = tmp_path/'model.txt'
    model.booster_.save_model(str(dest))
    restored = lgb.Booster(model_file=str(dest))
    assert restored.dump_model()['feature_infos']['Column_1']['values']
    np.testing.assert_allclose(restored.predict([[0, -1], [0, 1]], num_threads=1),
                               model.predict([[0, -1], [0, 1]]))
