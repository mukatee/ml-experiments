#unfortunately i do not remember why i had to set up seccomp as unconfined, but maybe better to try if it works without if repeating this image
docker run -p 8888:8888 --security-opt seccomp:unconfined --name amex -v /media/datamount/amex:/mystuff/data/amexdata -td amex
