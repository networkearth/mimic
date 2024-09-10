import json

import click

from mimic.log_odds.build_tfrecord import build_tfrecord as log_odds_build_tfrecord

@click.group()
def cli():
    pass

@cli.group()
def log_odds():
    pass

@log_odds.command()
@click.argument("config_path", required=True)
def build_tfrecord(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    log_odds_build_tfrecord(**config)
    