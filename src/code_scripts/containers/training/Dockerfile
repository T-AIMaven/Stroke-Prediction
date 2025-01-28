# | filename: Dockerfile
# | code-line-numbers: true

FROM python:3.10-slim

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    python3 \
    build-essential libssl-dev pkg-config libhdf5-dev

# Let's install the required Python packages from 
# the requirements.txt file.
COPY requirements.txt .
RUN pip install --user --upgrade pip
RUN pip3 install -r requirements.txt

# We are going to be running the training script
# as the entrypoint of this container.
COPY train.py /opt/ml/code/train.py
ENV SAGEMAKER_PROGRAM train.py
