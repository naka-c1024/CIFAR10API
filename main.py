import os
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
import urllib.error
import urllib.request
import json
from flask import Flask, jsonify, request, render_template

# PyTorchで定義されたネットワークのクラス
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # CNNの各層を定義
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(),
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

    def forward(self, xb):
        return self.network(xb)

net = Net()

# 画像分類のクラス
class Classifier:
    def __init__(self, model_path):
        # モデルの読み込み
        self.net = Net()
        self.net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.net.eval()

        # クラス名の定義
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.ja_classes = ['飛行機', '自動車', '鳥', '猫', '鹿', '犬', 'カエル', '馬', '船', 'トラック']

        # 画像の前処理
        self.preprocess = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # 画像を入力として受け取り、分類結果とスコアを返す
    def predict(self, image):
        inputs = self.preprocess(image).unsqueeze(0)
        outputs = self.net(inputs)
        _, predicted = torch.max(outputs, 1)

        classification_results = self.classes[predicted]
        predict_score = float(F.softmax(outputs, dim=1)[0][predicted][0])

        return classification_results, predict_score

# 画像をリクエストから取得する関数
def get_image_from_request(request):
    try:
        if request.is_json and 'image_uri' in request.json:
            image_uri = request.json['image_uri']
            return Image.open(urllib.request.urlopen(image_uri))
        elif 'image_file' in request.files:
            image_file = request.files['image_file']
            return Image.open(image_file).convert('RGB')
    except (UnidentifiedImageError, ValueError, urllib.error.HTTPError, urllib.error.URLError):
        return None
    return None

app = Flask(__name__)
classifier = Classifier("cifar_net.pth")

@app.route("/", methods=["GET", "POST"])
def index():
    # POST メソッドでアクセスされた場合
    if request.method == "POST":
        # 画像のURLまたはファイルを取得
        image_uri = request.form.get("image_uri")
        image_file = request.files.get("image_file")
        # 画像のURLまたはファイルが指定されていない場合、エラーを表示
        if not image_uri and not image_file:
            return render_template("index.html", error="画像のURLまたはファイルを指定してください。")
        if image_uri and image_file:
            return render_template("index.html", error="画像のURLとファイルの両方を指定することはできません。")

        try:
            # 画像を開く
            if image_uri:
                image = Image.open(urllib.request.urlopen(image_uri))
            elif image_file:
                image = Image.open(image_file).convert('RGB')
        except (UnidentifiedImageError, ValueError, urllib.error.HTTPError, urllib.error.URLError):
            # 無効な画像データの場合、エラーを表示
            return render_template("index.html", error="無効な画像dataです。")

        # 画像の分類結果とスコアを取得
        classification_results, predict_score = classifier.predict(image)
        # 分類結果を日本語に変換
        classification_results = classifier.ja_classes[classifier.classes.index(classification_results)]
        # スコアをパーセント表記に変換
        predict_score = round(predict_score * 100, 2)
        return render_template("index.html", classification_results=classification_results, predict_score=str(predict_score))

    # GET メソッドでアクセスされた場合
    return render_template("index.html")


# API機能
@app.route('/predict', methods=['POST'])
def predict_route():
    # リクエストから画像を取得
    image = get_image_from_request(request)
    # 画像が無効または提供されていない場合、エラーを返す
    if image is None:
        return jsonify({'error': 'Invalid or no image provided.'}), 400

    # 画像の分類結果とスコアを取得
    classification_results, predict_score = classifier.predict(image)

    # 分類結果とスコアをJSON形式で返す
    json_data = {
        "predictions": [
            {
                "classification_results": [classification_results],
                "score": [predict_score]
            }
        ]
    }
    return jsonify(json_data), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
