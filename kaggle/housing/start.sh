docker run -p 8890:8888 -p 40000:40000 --gpus all --name housing -v ./notebooks:/mystuff/notebooks -v /media/datadisk/kaggle-housing:/mystuff/data -td housing

