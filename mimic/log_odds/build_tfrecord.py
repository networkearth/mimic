import os
import json

import tensorflow as tf 
import pandas as pd
import boto3

import haven.db as db

def read_from_athena(database, table, partition, total_partitions, train):
    os.environ["HAVEN_DATABASE"] = database
    sql = f"""
    select 
        *
    from 
        {table}
    where 
        _decision % {total_partitions} = {partition}
        and {"" if train else "not"} _train
    """
    return db.read_data(sql)

def collapse_choices_row(max_choices, features, missing_values_map, group):
    new_row = {}
    for i, (_, row) in enumerate(group.iterrows()):
        for feature in features:
            if row['_selected']:
                new_row['_selected'] = i
            new_row[f"{feature}_{i}"] = row[feature]

    if i < max_choices - 1:
        for j in range(i + 1, max_choices):
            for feature in features:
                new_row[f"{feature}_{j}"] = missing_values_map[feature]

    return pd.DataFrame([new_row])

def collapse_choices(max_choices, features, missing_values_map, dataframe):
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

def serialize_row(max_choices, features, row):
    feature = {
        "_selected": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[int(row["_selected"])])
        ),
    }
    for i in range(max_choices):
        current_feature = {
            f"{feature}_{i}": tf.train.Feature(
                float_list=tf.train.FloatList(value=[row[f"{feature}_{i}"]])
            )
            for feature in features
        }
        feature.update(current_feature)
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecord(max_choices, features, dataframe, output_path):
    with tf.io.TFRecordWriter(output_path) as writer:
        for serialized_row in dataframe.apply(lambda row: serialize_row(max_choices, features, row), axis=1):
            writer.write(serialized_row)

def write_to_s3(space, dataset, partition, train, tfrecord_path):
    s3 = boto3.client('s3')
    bucket = f"{space}-tfrecords"
    key = f"{dataset}/{'train' if train else 'test'}/partition={partition}/data.tfrecord"
    s3.upload_file(tfrecord_path, bucket, key)

def build_tfrecord(database, table, partition, total_partitions, train, max_choices, features, missing_values_map, space, dataset):
    data = read_from_athena(database, table, partition, total_partitions, train)
    collapsed = collapse_choices(max_choices, features, missing_values_map, data)
    tfrecord_path = f"{space}_{dataset}_{partition}.tfrecord"
    write_tfrecord(max_choices, features, collapsed, tfrecord_path)
    write_to_s3(space, dataset, partition, train, tfrecord_path)
    os.remove(tfrecord_path)

        
