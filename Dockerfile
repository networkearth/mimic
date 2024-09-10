FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install build-essential wget git vim curl

RUN apt-get -y install python3 python-is-python3
RUN apt-get -y install python3-pip

RUN apt-get install -y npm
RUN npm install -g aws-cdk
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash


# https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
# chose what works for an M1 Mac
RUN apt-get install -y unzip
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip" 
RUN unzip awscliv2.zip
RUN ./aws/install
RUN rm -rf awscliv2.zip aws
