import os
import boto3

def handler(event, context):
    client = boto3.client('batch', 'us-east-1')

    job_name = "test-lambda-job"
    job_queue = "mimic-log-odds-job-queue"
    job_definition = "mimic-log-odds-build-tfrecords"

    print(f"Submitting job named '{job_name}' to queue '{job_queue}' "
                f"with definition '{job_definition}'")

    response = client.submit_job(
        jobName=job_name,
        jobQueue=job_queue,
        jobDefinition=job_definition,
    )

    print(response)

    return {
        'statusCode': 200,
        'body': 'Hello, World!'
    }