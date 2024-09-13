import os
import json
import hashlib
from copy import copy 

import boto3

def handler(event, context):
    s3 = boto3.client('s3')
    # read file from s3
    experiment_name = event["experiment_name"]
    config_key = f"{experiment_name}/config.json"
    response = s3.get_object(Bucket="mimic-log-odds-models", Key=config_key)
    config = json.loads(response['Body'].read().decode('utf-8'))

    for model in config["models"]:
        run_config = copy(config)
        run_config["model"] = model
        del run_config["models"]
        run_id = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode("utf-8")).hexdigest()
        run_config["run_id"] = run_id
        key = f"{experiment_name}/{run_id}/config.json"
        s3.put_object(Bucket="mimic-log-odds-models", Key=key, Body=json.dumps(run_config))

    return {
        'statusCode': 200
    }
