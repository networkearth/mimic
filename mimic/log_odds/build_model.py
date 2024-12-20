import os
import json
import shutil
import sys
from functools import partial

import boto3
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd

import haven.db as db


def setup_experiment(config_path, layers_path):
    with open(config_path, 'r') as fh:
        config = json.load(fh)

    experiment_name = config["experiment_name"]
    bucket_name = "-".join([config["space"], "models"])

    config_key = "/".join([experiment_name, "config.json"])
    layers_key = "/".join([experiment_name, "layers.py"])

    with open(layers_path, 'r') as fh:
        layers = fh.read()

    s3 = boto3.client('s3')
    s3.put_object(Bucket=bucket_name, Key=config_key, Body=json.dumps(config))
    s3.put_object(Bucket=bucket_name, Key=layers_key, Body=layers)


# pylint: disable=redefined-builtin
def split_data(N, features, data):
    """
    Inputs:
    - N: int, number of choices
    - features: list of strings, names of features
    - data: dict, data loaded from tfrecord file

    Splits the data into inputs and labels for the model
    """
    inputs = {}
    for i in range(N):
        input = tf.stack(
            [tf.cast(data[f"{feature}_{i}"], tf.float32) for feature in features]
        )
        inputs[f"input_{i}"] = input
    label = to_categorical(data["_selected"], num_classes=N)
    return inputs, label


def load_data(data_dir, N, features, batch_size, shuffle_buffer_size):
    """
    Inputs:
    - data_dir: str, path to directory containing tfrecord files
    - N: int, number of choices
    - features: list of strings, names of features
    - batch_size: int, batch size
    - shuffle_buffer_size: int, size of buffer for shuffling data

    Returns a tf.data.Dataset object containing the data
    """
    feature_description = {
        "_selected": tf.io.FixedLenFeature([], tf.int64),
    }
    for i in range(N):
        for feature in features:
            feature_description[f"{feature}_{i}"] = tf.io.FixedLenFeature(
                [], tf.float32
            )

    def _parse_function(proto):
        return tf.io.parse_single_example(proto, feature_description)

    tfrecord_files = []
    for path in os.listdir(data_dir):
        if path.endswith(".tfrecord"):
            tfrecord_files.append(os.path.join(data_dir, path))

    data = tf.data.TFRecordDataset(tfrecord_files)
    data = data.map(_parse_function)
    data = data.map(partial(split_data, N, features))
    data = data.shuffle(buffer_size=shuffle_buffer_size)
    data = data.batch(batch_size=batch_size)
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return data


def build_model(config, N, features, layers, final_activation="linear"):
    """
    Inputs:
    - N: int, number of choices
    - features: list of strings, names of features
    - layers: list of keras layers
    - final_activation: str, activation function for final layer

    Returns a keras model
    """
    layers.append(Dense(1, activation=final_activation))
    inputs = [Input(shape=(len(features),), name=f"input_{i}") for i in range(N)]
    outcomes = []
    for input in inputs:
        last_layer = input
        for layer in layers:
            last_layer = layer(last_layer)
        outcomes.append(last_layer)

    outcomes = concatenate(outcomes)

    output_layer = Dense(N, activation="softmax")
    output = output_layer(outcomes)
    output_layer.set_weights([np.eye(N), np.zeros(N)])
    output_layer.trainable = False

    model = Model(inputs=inputs, outputs=output)
    optimizer = Adam(**config['model'].get('optimizer_kwargs', {}))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy")

    return model, layers


def build_export_model(features, layers):
    input = Input(shape=(len(features),), name=f"input")
    last_layer = input 
    for layer in layers:
        last_layer = layer(last_layer)
    return Model(inputs=input, outputs=last_layer)

def pull_run_config(experiment_name, run_id):
    bucket_name = "mimic-log-odds-models"
    config_key = f"{experiment_name}/{run_id}/config.json"
    layers_key = f"{experiment_name}/layers.py"

    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=config_key)
    config = response['Body'].read().decode('utf-8')
    with open("config.json", "w") as fh:
        fh.write(config)

    response = s3.get_object(Bucket=bucket_name, Key=layers_key)
    layers = response['Body'].read().decode('utf-8')
    with open("layers.py", "w") as fh:
        fh.write(layers)

    return config


def pull_training_data(config_path):
    with open(config_path, 'r') as fh:
        config = json.load(fh)

    bucket_name = "mimic-log-odds-tfrecords"
    dataset = config["dataset"]

    s3 = boto3.client('s3')
    
    response = s3.list_objects_v2(
        Bucket=bucket_name,
        Prefix=f'{dataset}/train/'
    )

    if os.path.exists('train'):
        shutil.rmtree('train')
    os.mkdir('train')
    for i, content in enumerate(response.get('Contents', [])):
        key = content['Key']
        s3.download_file(bucket_name, key, f'train/part{i}-{key.split("/")[-1]}')

    response = s3.list_objects_v2(
        Bucket=bucket_name,
        Prefix=f'{dataset}/test/'
    )

    if os.path.exists('test'):
        shutil.rmtree('test')
    os.mkdir('test')
    for i, content in enumerate(response.get('Contents', [])):
        key = content['Key']
        s3.download_file(bucket_name, key, f'test/part{i}-{key.split("/")[-1]}')


def build_results(history):
    results = pd.DataFrame(history.history)
    results['epoch'] = results.index + 1
    return results


def train_model(config_path):
    with open(config_path, 'r') as fh:
        config = json.load(fh)

    sys.path.append(os.getcwd())
    from layers import LAYERS 

    batch_size = config["model"]["batch_size"]
    epochs = config["model"]["epochs"]
    max_choices = config["max_choices"]
    features = config["features"]
    layers = [LAYERS[layer]() for layer in config["model"]["layers"]]

    train = load_data('train', max_choices, features, batch_size=batch_size, shuffle_buffer_size=10000)
    test = load_data('test', max_choices, features, batch_size=batch_size, shuffle_buffer_size=10000)

    model, layers = build_model(config, max_choices, features, layers)

    history = model.fit(train, validation_data=test, epochs=epochs)
    results = build_results(history)
    results['experiment_name'] = config['experiment_name']
    results['run_id'] = config['run_id']

    os.environ["HAVEN_DATABASE"] = config["database"]
    db.write_data(results, config["table"], ['experiment_name', 'run_id'])

    model = build_export_model(features, layers)

    model.save('model.keras')

    s3 = boto3.client('s3')
    bucket_name = "mimic-log-odds-models"
    key = f"{config['experiment_name']}/{config['run_id']}/model.keras"
    s3.upload_file('model.keras', bucket_name, key)





