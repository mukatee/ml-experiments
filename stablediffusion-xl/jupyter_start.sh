#need to use "allow-root" to run in docker and access:
#https://stackoverflow.com/questions/38830610/access-jupyter-notebook-running-on-docker-container
#jupyter lab --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='' --NotebookApp.password=''
#NotbookApp.token seems deprecated, and suggests ServerApp.token.
#which also seems deprecated and suggests some identityservice token, which is poorly documented.
#so going with this until it breaks
jupyter lab --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='??' --NotebookApp.password='??'

