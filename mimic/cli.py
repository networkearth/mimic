import json

import click

from mimic.log_odds.build_tfrecord import build_tfrecord as log_odds_build_tfrecord
from mimic.log_odds.build_model import (
    setup_experiment,
    pull_run_config,
    pull_training_data,
    train_model,
)

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
@click.argument("layers_path", required=True)
def run_experiment(config_path, layers_path):
    setup_experiment(config_path, layers_path)

@log_odds.command()
@click.argument("experiment_name", required=True)
@click.argument("run_id", required=True)
def run_train_model(experiment_name, run_id):
    pull_run_config(experiment_name, run_id)
    pull_training_data("config.json")
    train_model("config.json")
    