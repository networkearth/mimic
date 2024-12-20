# mimic
Building log-odds models happens in roughly three steps:

1. **Data Preparation**: This step involves preparing the data for the model. This includes cleaning the data, transforming the data, and splitting the data into training and testing sets.
2. **Model Building**: This step involves building the model. This includes selecting the model, training the model, and evaluating the model.
3. **Model Deployment**: This step involves deploying the model. This includes saving the model, deploying the model, and monitoring the model.

The purpose of `mimic` is to deal with the latter two steps for you. Specifically, once you've prepared your
data in the right way, `mimic` will build the model for you and handle inferences as well. This is done using 
the `mimic-log-odds` application set built in AWS. 

Why the cloud? Because building log-odds models can be computationally expensive. There will be loads of 
decisions to train from (and infer on) and many different model architectures and hyperparameter sets you'll want to try. `mimic` allows you to parallelize all of this so that you can spend more of your time on the data preparation and model selection steps - the bits where a human brain is most valuable.

## First Time Setup

```bash
pip install .
pip install -r requirements.txt
```

## Logging Into AWS

```bash
aws sso configure
export AWS_PROFILE=<profile>
watercycle login <profile>
```

## Using `mimic-log-odds`

`mimic-log-odds` is a command line application that allows you to build and deploy log-odds models in AWS. There are three main commands you'll use:

### 1) Building Training Sets

```bash
mimic log-odds build-dataset <config_path>
```

The purpose of this command is to take a dataset  in `haven` and, in a distributed fashion, 
convert it to the rather awkward format required by log-odds models and save it as tfrecrods that can be used to train models using `keras`. 

Its one argument is a path to a config that should look something like:

```json
{
    "database": "haven",
    "table": "example_log_odds_features",
    "train_partitions": 2,
    "test_partitions": 1,
    "max_choices": 3,
    "features": ["size", "age"],
    "missing_values_map": {
        "size": -1,
        "age": -1
    },
    "space": "mimic-log-odds",
    "dataset": "example-dataset"
}
```

- `database`: the name of the database you'll be pulling from.
- `table`: the name of the table you'll be pulling from.
- `train_partitions`: the number of workers to use to build the training set.
- `test_partitions`: the number of workers to use to build the testing set.
- `max_choices`: the maximum number of choices per decision in the dataset.
- `features`: the features to include in the dataset.
- `missing_values_map`: a map of feature names to the value that represents a missing value.
- `space`: the space to save the dataset in.
- `dataset`: the name of the dataset (this will allow you to identify it later).

Your data will end up in an s3 folder: `s3://<space>-tfrecords/<dataset>/`.

So what format does the `table` need to be in?

Well in addition to your feature columns (which should be normalized at this point), you'll want a few other features:

- `_individual` (`int`): a unique identifier for each individual making decisions (your training and testing data should not include the same individuals).
- `_decision` (`int`): a unique identifier for each decision made by an individual. Partitions will be built on this value. 
- `_choice` (`int`): a unique identifier for each choice available in a specific decision. 
- `_selected` (`bool`): a boolean indicating whether the choice was selected by the individual for this decision. 
- `_train` (`bool`): a boolean indicating whether the decision should be used for training or testing.

### 2) Building Models

```bash
mimic log-odds run-experiment <config_path> <layers_path>
```

The purpose of this command is to build a log-odds model using a dataset you've built. It allows you to build several at a time in parallel so that you can do hyperparameter tuning. It also pushes the statistics of the training to a database so that you can understand how the models are performing.

The first argument should be the path to a config that looks like:

```json
{
    "experiment_name": "test-experiment",
    "space": "mimic-log-odds",
    "dataset": "example-dataset",
    "max_choices": 3,
    "features": ["size", "age"],
    "database": "haven",
    "table": "mimic_log_odds_results",
    "models": [
        {
            "batch_size": 100,
            "epochs": 10,
            "layers": ["D16", "D32", "D16"],
            "optimizer": "Adam",
            "optimizer_kwargs": {
                "learning_rate": 0.002
            }
        },
        {
            "batch_size": 100,
            "epochs": 10,
            "layers": ["D16", "D32", "D16", "D8"]
        }
    ]
}
```

- `experiment_name`: the name of the experiment.
- `space`: the space the dataset is saved in.
- `dataset`: the name of the dataset.
- `max_choices`: the maximum number of choices per decision in the dataset.
- `features`: the features to include in the dataset (can be a subset of the features in the dataset).
- `database`: the name of the database you'll be pulling from and pushing to.
- `table`: the name of the table you'll be pushing results to.
- `models`: a list of models to build. Each model should have:
    - `batch_size`: the batch size to use for training.
    - `epochs`: the number of epochs to train for.
    - `layers`: a list of layers to use in the model. These refer to the `LAYERS` in the second input to the command. 
    - `optimizer`: (optional) the name of the optimizer you want to use in `tf.keras.optimizers`. Defaults to `Adam`
    - `optimizer_kwargs`: (optional) a set of kwargs to pass to the optimizer. 

The second argument should be the path to a file that looks like:

```python
from tensorflow.keras.layers import Dense

LAYERS = {
    "D8": lambda: Dense(8, activation='relu'),
    "D16": lambda: Dense(16, activation='relu'),
    "D32": lambda: Dense(32, activation='relu'),
}
```

This file should define a dictionary called `LAYERS` where the keys are the names of the layers and the values are functions that return the layers.

To find the results of your experiment, you can query the `table` you specified in the config. You'll get the following columns:

- `loss`: the loss of the model on the training set.
- `val_loss`: the loss of the model on the testing set.
- `experiment_name`: the name of the experiment.
- `run_id`: a unique identifier for the run.
- `epoch`: the training epoch. 

If you wish to see the model itself or its specific configuration you can go to: `s3://<space>-models/<experiment_name>/<run_id>/`.

### 3) Making Inferences

```bash
mimic log-odds run-batch-infer <config_path>
```

The purpose of this command is to make inferences on a dataset using a model you've built. 

It's only input is a path to a config that should look like:

```json
{
    "database": "haven",
    "table": "example_log_odds_features",
    "train_partitions": 2,
    "test_partitions": 1,
    "features": ["size", "age"],
    "space": "mimic-log-odds",
    "experiment_name": "test-experiment",
    "run_id": "07c49bc09b5b47fca0fbcd0aba35f414ce6a129a12ee7f0310df016f084cda7f",
    "upload_table": "example_log_odds_batch_infer"
}
```

- `database`: the name of the database you'll be pulling from and pushing to.
- `table`: the name of the table you'll be pulling from. (this should have the same format is a table you'd use to build a dataset).
- `train_partitions`: the number of workers to use to infer on the training set.
- `test_partitions`: the number of workers to use to infer on the testing set.
- `features`: the features to include in the dataset.
- `space`: the space the dataset is saved in.
- `experiment_name`: the name of the experiment.
- `run_id`: the run id of the model you want to use.
- `upload_table`: the name of the table you'll be pushing results to.

The results of the inference will be pushed to the `upload_table` you specified. The table will include the `_individual`, `_decision`, `_choice`, and `_train` columns noted in "Building Training Sets", a `probability` column that represents the probability of the choice being selected (as predicted by the model), and `experiment_name`, `run_id`, and `_partition` columns. In addition there will also now be a `log_odds` column that represents the log-odds of the choice being selected (as predicted by the model) and an `odds` column that represents the odds of the choice being selected (as predicted by the model).

**BEWARE!** the uploaded data is partitioned by `experiment_name`, `run_id`, `_train`, and `_partition`. This means that a rerun that has a different number of partitions may not overwrite the previous results. 

### 4) Building Contrasts

Sometimes you're `max_choices` leads to some serious class imbalance (lots of zeros). In these cases, you may want to build a contrast set that has a more balanced distribution of classes. 

```bash
mimic log-odds build-contrast <config_path>
```

This will build doublets of choices sampled from the old dataset so that 
each individual and choice is represented evenly. 

The config should look like:

```json
{
    "database": "haven",
    "source_table": "example_log_odds_features",
    "train_partitions": 2,
    "test_partitions": 1,
    "destination_table": "example_log_odds_contrasts",
    "decisions_per_individual": 2, 
    "alternatives_per_decision": 2
}
```

- `database`: the name of the database you'll be pulling from and pushing to.
- `source_table`: the name of the table you'll be pulling from (note this should be a features table
like that required for building a dataset).
- `train_partitions`: the number of workers to use to build the training set.
- `test_partitions`: the number of workers to use to build the testing set.
- `destination_table`: the name of the table you'll be pushing the contrasts to. This table will look exactly
like the source table but with doubles for each decision and `_old_decision` and `_old_choice` columns that reference back to the source table.
- `decisions_per_individual`: the number of decisions to sample per individual.
- `alternatives_per_decision`: the number of alternatives to sample per decision. Note that the 
total number of new decisions will be `decisions_per_individual * alternatives_per_decision`. 

The resulting table can be used to build a dataset like normal but will be guaranteed to have an even number of selected and unselected choices (as they are all doublets).

## Components

The best way to learn about the components of `mimic` is to look at the code. However, here's a brief overview:

1. `mimic-log-odds-compute` - the compute environment where batch jobs are run.
2. `mimic-log-odds-job-queue` - the job queue where batch jobs are submitted.
3. `mimic-log-odds-vpc` - the vpc where the compute environment is run.
4. `mimic-log-odds-execution-role` - the execution role used by the batch jobs and lambdas.
5. `mimic-log-odds-lambda`: the s3 bucket where the lambda code is stored.
6. `mimic-log-odds-models`: the s3 bucket where the models are stored.
7. `mimic-log-odds-tfrecords`: the s3 bucket where the tfrecords are stored.
8. `mimic-log-odds-build-tfrecords`: the batch job that builds a partition's worth of tfrecords.
9. `mimic-log-odds-run-train-model`: the batch job that trains a single model.
10. `mimic-log-odds-batch-infer-partition`: the batch job that makes inferences on a partition's worth of dataset.
11. `mimic-log-odds-build-tfrecords`: the lambda that kicks off the build tfrecords jobs for all the partitions.
12. `mimic-log-odds-run-experiment`: the lambda that kicks off the run train model jobs for all the models.
13. `mimic-log-odds-batch-infer`: the lambda that kicks off the batch infer partition jobs for all the partitions.

## Deploying `mimic-log-odds`

Thanks to `watercycle` it's as easy as navigating to each component directory and running:

```bash
watercycle deploy <deployment>
```
