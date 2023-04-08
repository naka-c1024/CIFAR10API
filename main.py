import os
import io
import json

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image

from flask import Flask, jsonify, request
import urllib.request

# NNの定義
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# モデルの読み込み
# net = Net() # インスタンス化
PATH = "cifar_net.pth"

net = torch.load(PATH, torch.device('cpu'))

# net.load_state_dict(torch.load(PATH))
# net = net.cpu() # 元々GPUで作成したものなのでCPUに変換する
# net.eval()

# クラス名の定義
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 画像の前処理
preprocess = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

app = Flask(__name__)

@app.route("/")
def index():
    json_data = {
        "predictions": [
            {
                "classification_results": [
                    "cat"
                ],
                "score": [
                    0.8342
                ]
            }
        ]
    }
    return jsonify(json_data), 200

@app.route('/predict', methods=['POST'])
def predict():
    '''
    # 画像を受け取ります。
    if 'image_url' in request.json:
        # 画像がURLで与えられた場合は、画像をダウンロードします。
        url = request.json['image_url']
        image = urllib.request.urlopen(url).read()
        image = tf.io.decode_image(image, channels=3)
    elif 'image_file' in request.files:
        # 画像がファイルで与えられた場合は、画像を読み込みます。
        image_file = request.files['image_file']
        image = tf.io.decode_image(image_file.read(), channels=3)
    else:
        # 画像が与えられていない場合は、エラーを返します。
        return jsonify({'error': 'No image provided.'}), 400
    '''

    '''
    image_file = request.files['image_file']
    image = Image.open(io.BytesIO(image_file)).convert('RGB')

    # 画像の前処理
    inputs = preprocess(image).unsqueeze(0)

    # 予測
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)

    # 結果をJSON形式で返す
    return jsonify({
        'predictions': [{
            'classification_results': [classes[predicted[0]]],
            'score': [float(F.softmax(outputs, dim=1)[0][predicted[0]])]
        }]
    }), 200
    '''

    json_data = {
        "predictions": [
            {
                "classification_results": [
                    "cat"
                ],
                "score": [
                    0.8342
                ]
            }
        ]
    }
    print(request.get_json())
    return jsonify(json_data), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
