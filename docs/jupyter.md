Suppose you run with:
`docker run -d --cap-add sys_ptrace -p127.0.0.1:2222:22 -p127.0.0.1:8888:8888 --name qu-pytorch rustil/qu-pytorch:ssh`

then you could ssh into the container and start a jupyter notebook server:

`ssh -p 2222 -t user@localhost bash` (the pwd is likely "password")

switch to the pytorch environment (which includes jupyter):

`conda activate`

and finally start a notebook server:

`jupyter notebook --ip=0.0.0.0 --no-browser`

then copy the url/token into your browser and enjoy. Here you are fully contained within the container, so maybe think
about `--mount`ing any volumes that you'd like to save to / read from. 