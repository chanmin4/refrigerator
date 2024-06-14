# Use the official miniconda3 image as a base
FROM continuumio/miniconda3

# Create a new conda environment with Python 3.10.14
RUN conda create -n final_env python=3.10.14 -y

# Activate the environment and set it as the default
SHELL ["conda", "run", "-n", "final_env", "/bin/bash", "-c"]

# Copy the requirements file into the container
COPY requirements-linux.txt /tmp/requirements-linux.txt

# Install the required packages
RUN conda install -n final_env pip && pip install -r /tmp/requirements.txt

# Set the environment variable to use the new environment
ENV PATH /opt/conda/envs/final_env/bin:$PATH
