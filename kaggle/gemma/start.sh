docker run -p 8890:8888 -p 40000:40000 --gpus all --name gemma -v ./notebooks:/mystuff/notebooks -v /media/datadisk2/llm/:/mystuff/llm -v /media/datadisk1/kaggle-gemma:/mystuff/data -td gemma

