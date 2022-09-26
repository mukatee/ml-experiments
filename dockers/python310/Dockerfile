FROM tensorflow/tensorflow:latest-gpu

ENV TZ=Europe/Helsinki

#this appears to be setting timezone for Docker image, along with the python repos etc
RUN	ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone &&\
	apt install software-properties-common; add-apt-repository ppa:deadsnakes/ppa &&\
	apt update -y; apt upgrade -y; apt install python3.10 -y

RUN rm -f /usr/local/bin/python && ln -s /usr/bin/python3.10 /usr/local/bin/python

RUN apt install python3.10-distutils -y

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

#RUN python3.10 -m pip install --upgrade pip

RUN python3.10 -m pip install --upgrade setuptools pip pipenv

#RUN python3.9 -m pip install pipenv jupyter

ENV PROJECT_DIR /mystuff

RUN mkdir ${PROJECT_DIR}

WORKDIR ${PROJECT_DIR}

#pipfile lock not needed when pipenv install --system?
COPY Pipfile Pipfile.lock jupyter_start.sh ${PROJECT_DIR}/

RUN pipenv install --system --deploy

