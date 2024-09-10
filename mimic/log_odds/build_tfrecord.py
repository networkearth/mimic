import tensorflow as tf 
import pandas as pd

def collapse_choices_row(max_choices, features, missing_values_map, group):
    new_row = {}
    for i, (_, row) in enumerate(group.iterrows()):
        for feature in features:
            if row['_selected'] == 1:
                new_row['_selected'] = i
            new_row[f"{feature}_{i}"] = row[feature]

    if i < max_choices - 1:
        for j in range(i + 1, max_choices):
            for feature in features:
                new_row[f"{feature}_{j}"] = missing_values_map[feature]

    return pd.DataFrame([new_row])

def collapse_choices(features, missing_values_map, dataframe):
    max_choices = dataframe.groupby(['_decision', '_individual'])[['_selected']].count().max().values[0]
    collapsed = (
        dataframe.groupby(['_decision', '_individual'])
        .apply(
            lambda g: collapse_choices_row(
                max_choices,
                features,
                missing_values_map,
                g
            )
        )
        .reset_index()
    )
    return collapsed[[c for c in collapsed.columns if c != 'level_2']]

        
