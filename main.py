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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

class Classifier:
    def __init__(self, model_path):
        self.net = Net()
        self.net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.net.eval()

        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.ja_classes = ['飛行機', '自動車', '鳥', '猫', '鹿', '犬', 'カエル', '馬', '船', 'トラック']

        self.preprocess = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def predict(self, image):
        inputs = self.preprocess(image).unsqueeze(0)
        outputs = self.net(inputs)
        _, predicted = torch.max(outputs, 1)

        classification_results = self.classes[predicted]
        predict_score = float(F.softmax(outputs, dim=1)[0][predicted][0])

        return classification_results, predict_score

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
        image_uri = request.form.get("image_uri")
        image_file = request.files.get("image_file")
        if not image_uri and not image_file:
            return render_template("index.html", error="画像のURLまたはファイルを指定してください。")

        try:
            if image_uri:
                image = Image.open(urllib.request.urlopen(image_uri))
            elif image_file:
                image = Image.open(image_file).convert('RGB')
        except (UnidentifiedImageError, ValueError, urllib.error.HTTPError, urllib.error.URLError):
            return render_template("index.html", error="無効な画像dataです。")

        classification_results, predict_score = classifier.predict(image)
        classification_results = classifier.ja_classes[classifier.classes.index(classification_results)]
        predict_score = round(predict_score * 100, 2)
        return render_template("index.html", classification_results=classification_results, predict_score=str(predict_score))

    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict_route():
    image = get_image_from_request(request)
    if image is None:
        return jsonify({'error': 'Invalid or no image provided.'}), 400

    classification_results, predict_score = classifier.predict(image)

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
