docker run -p 8890:8888 -p 40000:40000 --gpus all --name starchat-b -v ./notebooks:/mystuff/notebooks -v /media/datadisk2/llm/:/mystuff/data -td llm-torch

