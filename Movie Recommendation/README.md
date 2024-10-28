## Build and run the Docker container using:

```bash
docker build -t movie-recommender .

docker run -p 9000:9000 movie-recommender 
```

> You can use any port, I have used 9000 just to set an example for understanding the port-mapping concept in docker.