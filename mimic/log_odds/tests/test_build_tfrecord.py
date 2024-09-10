import os

import pandas as pd
from pandas.testing import assert_frame_equal

from mimic.log_odds.build_tfrecord import (
    collapse_choices,
    serialize_row,
    write_tfrecord,
)

def test_collapse_choices():
    data = pd.DataFrame([
        {'_individual': 'a', '_decision': 0, '_selected': False, 'feature1': 0.3, 'feature2': 0.4},
        {'_individual': 'a', '_decision': 0, '_selected': True, 'feature1': 0.6, 'feature2': 0.8},
        {'_individual': 'a', '_decision': 1, '_selected': True, 'feature1': 0.9, 'feature2': 0.3},
        {'_individual': 'a', '_decision': 1, '_selected': False, 'feature1': 0.2, 'feature2': 0.1},
        {'_individual': 'a', '_decision': 1, '_selected': False, 'feature1': 0.6, 'feature2': 0.4},
    ])

    max_choices = 3
    features = ['feature1', 'feature2']
    missing_values_map = {feature: -1.0 for feature in features}

    result = collapse_choices(max_choices, features, missing_values_map, data)
    expected = pd.DataFrame([
        {'_individual': 'a', '_decision': 0, '_selected': 1, 'feature1_0': 0.3, 'feature2_0': 0.4, 'feature1_1': 0.6, 'feature2_1': 0.8, 'feature1_2': -1.0, 'feature2_2': -1.0},
        {'_individual': 'a', '_decision': 1, '_selected': 0, 'feature1_0': 0.9, 'feature2_0': 0.3, 'feature1_1': 0.2, 'feature2_1': 0.1, 'feature1_2': 0.6, 'feature2_2': 0.4},
    ])

    assert set(result.columns) == set(expected.columns)
    assert_frame_equal(result[result.columns], expected[result.columns])

def test_serialize_row():
    data = pd.DataFrame([
        {'_individual': 'a', '_decision': 0, '_selected': 1, 'feature1_0': 0.3, 'feature2_0': 0.4, 'feature1_1': 0.6, 'feature2_1': 0.8, 'feature1_2': -1.0, 'feature2_2': -1.0},
        {'_individual': 'a', '_decision': 1, '_selected': 0, 'feature1_0': 0.9, 'feature2_0': 0.3, 'feature1_1': 0.2, 'feature2_1': 0.1, 'feature1_2': 0.6, 'feature2_2': 0.4},
    ])

    max_choices = 3
    features = ['feature1', 'feature2']

    data['serialized_record'] = data.apply(lambda row: serialize_row(max_choices, features, row), axis=1)

def test_write_tfrecord():
    data = pd.DataFrame([
        {'_individual': 'a', '_decision': 0, '_selected': 1, 'feature1_0': 0.3, 'feature2_0': 0.4, 'feature1_1': 0.6, 'feature2_1': 0.8, 'feature1_2': -1.0, 'feature2_2': -1.0},
        {'_individual': 'a', '_decision': 1, '_selected': 0, 'feature1_0': 0.9, 'feature2_0': 0.3, 'feature1_1': 0.2, 'feature2_1': 0.1, 'feature1_2': 0.6, 'feature2_2': 0.4},
    ])

    max_choices = 3
    features = ['feature1', 'feature2']
    output_path = 'test.tfrecord'

    try:
        write_tfrecord(max_choices, features, data, output_path)
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)

    