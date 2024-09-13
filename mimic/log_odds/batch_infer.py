import os

import boto3
import tensorflow.keras as keras
import numpy as np
import pandas as pd

from mimic.log_odds.build_tfrecord import read_from_athena, collapse_choices

import haven.db as db

def expand_choices(max_choices, data, features):
    columns_to_expand = features + ['_choice', 'probability']
    dataframes = []
    for i, _ in enumerate(range(max_choices)):
        collapsed_columns = [f"{col}_{i}" for col in columns_to_expand]
        dataframe = data[collapsed_columns + ['_individual', '_decision']].copy()
        dataframe = dataframe.rename(columns=dict(zip(collapsed_columns, columns_to_expand)))
        dataframes.append(dataframe)
    return pd.concat(dataframes)

def infer(model, data, features, max_choices, missing_values_map):
    missing_values_map['_choice'] = np.nan
    transformed = collapse_choices(max_choices, features + ['_choice'], missing_values_map, data)

    inputs = {}
    for i, _ in enumerate(range(max_choices)):
        sub_features = [f"{feature}_{i}" for feature in features]
        inputs[f"input_{i}"] = transformed[sub_features]
    
    predictions = model.predict(inputs)
    for i, _ in enumerate(range(max_choices)):
        transformed[f"probability_{i}"] = predictions[:, i]

    result = expand_choices(max_choices, transformed, features)
    result = result[~np.isnan(result['_choice'])]
    return result

def clear_data(database, table, experiment_name, run_id):
    os.environ["HAVEN_DATABASE"] = database
    db.delete_data(table, [{'experiment_name': experiment_name, 'run_id': run_id}])

def run_inference(
    database, table, partition, total_partitions, 
    train, max_choices, features, missing_values_map, 
    upload_table, space, experiment_name, run_id
):
    bucket_name = f"{space}-models"
    model_key = f"{experiment_name}/{run_id}/model.keras"
    s3 = boto3.client("s3")
    s3.download_file(bucket_name, model_key, "model.keras")

    model = keras.models.load_model("model.keras")

    data = read_from_athena(database, table, partition, total_partitions, train)

    results = infer(model, data, features, max_choices, missing_values_map)
    results['experiment_name'] = experiment_name
    results['run_id'] = run_id
    results['_partition'] = partition
    results['_train'] = train

    os.environ["HAVEN_DATABASE"] = database
    db.write_data(results, upload_table, ['experiment_name', 'run_id', '_train', '_partition'])
