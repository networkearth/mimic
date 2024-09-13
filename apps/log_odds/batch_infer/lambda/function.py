import os
import json
from copy import copy 

import boto3

def handler(event, context):
    client = boto3.client('batch', 'us-east-1')

    job_queue = "mimic-log-odds-job-queue"
    job_definition = "mimic-log-odds-batch-infer-partition"

    upload_table = event['upload_table'].replace('_', '-')
    
    base_config = copy(event)
    del base_config['train_partitions']
    del base_config['test_partitions']
    for train in [True, False]:
        total_partitions = event['train_partitions'] if train else event['test_partitions']
        for partition in range(total_partitions):
            config = copy(base_config)
            config['train'] = train
            config['partition'] = partition
            config['total_partitions'] = total_partitions

            command = json.dumps(config).replace(' ', '')
            client.submit_job(
                jobName=f"{job_definition}-{upload_table}-{train}-{partition}",
                jobQueue=job_queue,
                jobDefinition=job_definition,
                containerOverrides={
                    "command": [
                        command
                    ]
                }
            )

    return {
        'statusCode': 200
    }
