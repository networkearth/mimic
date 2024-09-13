import json

import click
import boto3

from mimic.log_odds.build_tfrecord import build_tfrecord as log_odds_build_tfrecord
from mimic.log_odds.build_model import (
    setup_experiment,
    pull_run_config,
    pull_training_data,
    train_model,
)
from mimic.log_odds.batch_infer import run_inference, clear_data

@click.group()
def cli():
    pass

@cli.command()
@click.option("--input-json", required=True)
@click.option("--output-path", required=True)
def drop_json(input_json, output_path):
    config = json.loads(input_json)
    with open(output_path, "w") as f:
        json.dump(config, f)

@cli.group()
def log_odds():
    pass

@log_odds.command()
@click.argument("config_path", required=True)
def build_tfrecord(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    log_odds_build_tfrecord(**config)

@log_odds.command()
@click.argument("config_path", required=True)
def build_dataset(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    client = boto3.client("lambda")
    client.invoke(
        FunctionName="mimic-log-odds-build-tfrecords",
        InvocationType="Event",
        Payload=json.dumps(config),
    )

@log_odds.command()
@click.argument("config_path", required=True)
@click.argument("layers_path", required=True)
def run_experiment(config_path, layers_path):
    setup_experiment(config_path, layers_path)
    with open(config_path, "r") as f:
        config = json.load(f)

    experiment_name = config["experiment_name"]
    client = boto3.client("lambda")
    client.invoke(
        FunctionName="mimic-log-odds-run-experiment",
        InvocationType="Event",
        Payload=json.dumps({"experiment_name": experiment_name}),
    )

@log_odds.command()
@click.argument("experiment_name", required=True)
@click.argument("run_id", required=True)
def run_train_model(experiment_name, run_id):
    pull_run_config(experiment_name, run_id)
    pull_training_data("config.json")
    train_model("config.json")

@log_odds.command()
@click.argument("config_path", required=True)
def run_batch_infer_partition(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    run_inference(**config)

@log_odds.command()
@click.argument("config_path", required=True)
def run_batch_infer(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    # need to clear because the number of 
    # partitions may have changed
    experiment_name = config["experiment_name"]
    run_id = config["run_id"]
    database = config["database"]
    table = config["table"]
    clear_data(database, table, experiment_name, run_id)
    
    client = boto3.client("lambda")
    client.invoke(
        FunctionName="mimic-log-odds-batch-infer",
        InvocationType="Event",
        Payload=json.dumps(config),
    )
