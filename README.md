# How to Run

#### Build Docker Image

```sh
docker build -t hello .
```

#### Create container and run
```sh
docker run --gpus all -it --name my_container hello
```
#### After done, copy all checkpoints from container to local path:
```sh
docker cp my_container:/app/checkpoints /path/to/local/checkpoints
```

