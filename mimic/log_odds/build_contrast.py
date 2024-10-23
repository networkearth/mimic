import os

import pandas as pd
import haven.db as db

def load_source_data(database, source_table, partition, total_partitions, train):
    # the basic unit here is an individual so we partition
    # on them
    sql = f'''
    select 
        *
    from 
        {source_table}
    where 
        _individual % {total_partitions} = {partition}
        and {"" if train else "not"} _train
    '''
    os.environ['HAVEN_DATABASE'] = database
    return db.read_data(sql)

def build_selections(data, decisions_per_individual):
    selections = data[data['_selected']]
    # we sample with replacement to get the desired number of decisions
    # per individual
    selections = (
        selections.groupby('_individual')
        .sample(n=decisions_per_individual, replace=True)
    )
    return selections

def build_alternatives(data, alternatives_per_decision, selections):
    alternatives = data[~data['_selected']]
    # we sample with replacement to get the desired number of alternatives
    # per decision
    alternatives = (
        alternatives.groupby(['_individual', '_decision'])
        .sample(n=alternatives_per_decision, replace=True)
    )
    # now we need to filter out the decisions that aren't
    # in the selections
    return (
        selections[['_individual', '_decision']]
        .merge(
            alternatives,
            on=['_individual', '_decision'],
            how='inner'
        )
    )

def combine(selections, alternatives, alternatives_per_decision):
    # we repeat the selections to match the number of alternatives
    selections = pd.concat([selections] * alternatives_per_decision)
    # we need to add a new _decision label to the alternatives
    # and selections first. We'll do this by sorting the
    # selections and alternatives by individual and decision
    # and then assigning the index as the new decision value
    selections = (
        selections.sort_values(['_individual', '_decision'])
        .reset_index(drop=True).reset_index()
        .rename(columns={'index': '_decision', '_decision': '_old_decision'})
    )
    alternatives = (
        alternatives.sort_values(['_individual', '_decision'])
        .reset_index(drop=True).reset_index()
        .rename(columns={'index': '_decision', '_decision': '_old_decision'})
    )
    combination = pd.concat([selections, alternatives])
    # now we need to create choice id's
    combination = (
        combination
        .reset_index(drop=True).reset_index()
        .rename(columns={'index': '_choice', '_choice': '_old_choice'})
    )
    return combination

def build_contrast_func(
    database, source_table, partition, total_partitions, train, destination_table,
    decisions_per_individual, alternatives_per_decision,
):
    data = load_source_data(database, source_table, partition, total_partitions, train)

    selections = build_selections(data, decisions_per_individual)
    alternatives = build_alternatives(data, alternatives_per_decision, selections)
    contrast = combine(selections, alternatives, alternatives_per_decision)

    os.environ['HAVEN_DATABASE'] = database
    contrast['partition'] = partition
    db.write_data(contrast, destination_table, ['partition'])


    



