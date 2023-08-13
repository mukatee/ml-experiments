FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
#FROM pytorch/pytorch:latest
#FROM ubuntu:22.04

ENV TZ=Europe/Helsinki

#this appears to be setting timezone for Docker image, along with the python repos etc
RUN	ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone &&\
	apt install software-properties-common jq nano; add-apt-repository ppa:deadsnakes/ppa&&\
	apt update -y; apt upgrade -y

#RUN	apt install pip

RUN pip install --upgrade setuptools pip pipenv

ENV PROJECT_DIR /mystuff

RUN mkdir ${PROJECT_DIR}

WORKDIR ${PROJECT_DIR}

RUN pip install --verbose numpy pandas matplotlib tqdm scipy scikit-learn transformers langchain huggingface_hub sentence_transformers accelerate bitsandbytes ipywidgets fastparquet
RUN pip install --verbose jupyterlab

COPY jupyter_start.sh ${PROJECT_DIR}/

COPY overrides.json /opt/conda/share/jupyter/lab/settings/

