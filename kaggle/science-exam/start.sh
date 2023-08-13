docker run -p 8890:8888 -p 40000:40000 --gpus all --name science-quiz -v ./notebooks:/mystuff/notebooks -v /media/datadisk2/llm/:/mystuff/data -v /media/datadisk2/wikipedia:/mystuff/wikipedia -td science-quiz

