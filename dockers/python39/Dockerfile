FROM tensorflow/tensorflow:latest-gpu

ENV TZ=Europe/Helsinki

#this appears to be setting timezone for Docker image, along with the python repos etc
RUN	ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone &&\
	apt update -y; apt install software-properties-common; add-apt-repository ppa:deadsnakes/ppa &&\
	python -m pip install --upgrade setuptools pip pipenv

ENV PROJECT_DIR /mystuff

RUN mkdir ${PROJECT_DIR}

WORKDIR ${PROJECT_DIR}

COPY Pipfile Pipfile.lock jupyter_start.sh ${PROJECT_DIR}/

RUN pipenv install --system --deploy

