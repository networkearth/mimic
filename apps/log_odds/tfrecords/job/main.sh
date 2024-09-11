#!/bin/bash

mimic drop-json --input-json $1 --output-path config.json
cat config.json
mimic log-odds build-tfrecord config.json