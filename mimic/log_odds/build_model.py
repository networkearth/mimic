import json

import boto3

def setup_experiment(config_path, layers_path):
    with open(config_path, 'r') as fh:
        config = json.load(fh)

    experiment_name = config["experiment_name"]
    bucket_name = "-".join([config["space"], "models"])

    config_key = "/".join([experiment_name, "config.json"])
    layers_key = "/".join([experiment_name, "layers.py"])

    s3 = boto3.client('s3')
    s3.put_object(Bucket=bucket_name, Key=config_key, Body=json.dumps(config))
    s3.put_object(Bucket=bucket_name, Key=layers_key, Body=json.dumps(layers_path))

