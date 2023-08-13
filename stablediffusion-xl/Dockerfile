FROM pytorch/pytorch:latest

ENV TZ=Europe/Helsinki

#this appears to be setting timezone for Docker image, along with the python repos etc
RUN	ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone &&\
	apt install software-properties-common jq nano; add-apt-repository ppa:deadsnakes/ppa&&\
	apt update -y; apt upgrade -y

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade setuptools pip pipenv

ENV PROJECT_DIR /mystuff

RUN mkdir ${PROJECT_DIR}

WORKDIR ${PROJECT_DIR}

RUN pip install --verbose numpy pandas matplotlib tqdm scipy scikit-learn transformers langchain huggingface_hub sentence_transformers\
                          accelerate ipywidgets safetensors invisible-watermark diffusers
RUN pip install --verbose jupyterlab

COPY jupyter_start.sh ${PROJECT_DIR}/

COPY overrides.json /opt/conda/share/jupyter/lab/settings/

