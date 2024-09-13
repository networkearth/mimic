import os
from random import random

import click
import pandas as pd

import haven.db as db

@click.command()
@click.option('--database', required=True)
@click.option('--table', required=True)
def build_dataset(database, table):
    os.environ["HAVEN_DATABASE"] = database

    db.drop_table(table)

    good_size, good_age = 10, 20
    bad_size, bad_age = 5, 10
    decisions = list(range(10))
    individuals = list(range(10))

    data = pd.DataFrame([
        {
            "_decision": decision,
            "_individual": individual,
            "size": good_size + random() * 5 if (decision + choice) % 2 == 0 else bad_size + random() * 5,
            "age": good_age + random() * 10 if (decision + choice) % 2 == 0 else bad_age + random() * 10,
            "_selected": True if (decision + choice) % 2 == 0 else False,
            "_train": True if individual % 3 != 0 else False
        }
        for choice in [0, 1]
        for decision in decisions
        for individual in individuals
    ])
    data = data.reset_index()
    data = data.rename(columns={"index": "_choice"})

    db.write_data(data, table, ["_train"])

if __name__ == '__main__':
    build_dataset()