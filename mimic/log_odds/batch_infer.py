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

def infer(model, data, features):
    data['log_odds'] = model.predict(data[features])
    data['odds'] = np.exp(data['log_odds'])
    data['sum_odds'] = data.groupby(['_individual', '_decision'])['odds'].transform('sum')
    data['probability'] = data['odds'] / data['sum_odds']
    del data['sum_odds']
    return data

def clear_data(database, table, experiment_name, run_id):
    os.environ["HAVEN_DATABASE"] = database
    db.delete_data(table, [{'experiment_name': experiment_name, 'run_id': run_id}])

def run_inference(
    database, table, partition, total_partitions, 
    train, features, upload_table, space, experiment_name, 
    run_id
):
    bucket_name = f"{space}-models"
    model_key = f"{experiment_name}/{run_id}/model.keras"
    s3 = boto3.client("s3")
    s3.download_file(bucket_name, model_key, "model.keras")

    model = keras.models.load_model("model.keras")

    data = read_from_athena(database, table, partition, total_partitions, train)

    results = infer(model, data, features)
    results['experiment_name'] = experiment_name
    results['run_id'] = run_id
    results['_partition'] = partition
    results['_train'] = train

    os.environ["HAVEN_DATABASE"] = database
    db.write_data(results, upload_table, ['experiment_name', 'run_id', '_train', '_partition'])
