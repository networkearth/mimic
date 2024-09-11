#!/bin/bash

mimic drop-json --input-json $1 --output-path config.json
mimic log-odds build_tfrecord config.json