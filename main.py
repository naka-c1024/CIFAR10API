import os

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
import urllib.request

import json
from flask import Flask, jsonify, request

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

# netの読み込み
PATH = "cifar_net.pth"
net = Net()
net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu'))) # GPUからCPUへ変更
net.eval()

# 分類するクラス名の定義
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 画像の前処理
preprocess = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    # 画像を受け取る
    if request.is_json and 'image_uri' in request.json:
        # 画像がURIで与えられた場合は、画像をダウンロード
        image_uri = request.json['image_uri']
        image = Image.open(urllib.request.urlopen(image_uri))
    elif 'image_file' in request.files:
        # 画像データが与えられた場合
        image_file = request.files['image_file']
        image = Image.open(image_file).convert('RGB')
    else:
        # 画像が与えられていない場合は、エラーを返す
        return jsonify({'error': 'No image provided.'}), 400

    # 画像の前処理
    inputs = preprocess(image).unsqueeze(0)

    # 予測
    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)

    classification_results = classes[predicted]
    predict_score = float(F.softmax(outputs, dim=1)[0][predicted][0])

    # 結果をJSON形式で返す
    json_data = {
        "predictions": [
            {
                "classification_results": [
                    classification_results
                ],
                "score": [
                    predict_score
                ]
            }
        ]
    }
    return jsonify(json_data), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
