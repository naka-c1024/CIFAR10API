```
docker build -t cifar10_api .
```

```
docker run -v $(pwd):/app -p 8080:8080 -it cifar10_api bash
```

飛行機のストレージURI
file:///app/sandbox/cifar10_images/test_2.png

'airplane'
https://cdn.hswstatic.com/gif/airplane-windows.jpg
'automobile'
'bird'
'cat'
https://speee-ad.akamaized.net/articles/fe7ee8fc1959cc7214fa21c4840dff0a/40c91a84d7116d25b6c26f19e7a213c7.jpg
'deer'
'dog'
'frog'
'horse'
'ship'
'truck'

URIの場合
curl -X POST -H 'Content-Type: application/json' -d '{"image_uri": "https://cdn.hswstatic.com/gif/airplane-windows.jpg"}' http://127.0.0.1:8080/predict

画像ファイルデータの場合
curl -X POST -F image_file=@/Users/nakashimayuto/CA_Tech_Lounge_CyberAgent/CIFAR10API/sandbox/cifar10_images/test_2.png http://127.0.0.1:8080/predict
