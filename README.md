```
docker build -t cifar10_api .
```

```
docker run -v $(pwd):/app -p 8080:8080 -it cifar10_api bash
```

```
curl http://127.0.0.1:8080/
curl -v http://127.0.0.1:8080/
```

```
curl -X POST -H "Content-Type: application/json" -d '{"image_url": "https://example.com/image.jpg"}' http://127.0.0.1:8080/predict
curl -X POST -H "Content-Type: application/json" -d '{"image_file": "/Users/nakashimayuto/CA_Tech_Lounge_CyberAgent/sandbox/cifar10_images/test_0.png"}' http://127.0.0.1:8080/predict
curl -X POST -H "Content-Type: application/json" -F file1=@/Users/nakashimayuto/CA_Tech_Lounge_CyberAgent/sandbox/cifar10_images/test_0.png http://127.0.0.1:8080/predict
```
