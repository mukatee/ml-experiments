#8889 is in host, exposing 8888 from container
#unfortunately i do not remember why i had to set up seccomp as unconfined, but maybe better to try if it works without if repeating this image
docker run -p 8889:8888 --gpus all --security-opt seccomp:unconfined --name amexgpu -v /media/datamount/amex:/mystuff/data/amexdata -td amexgpu
