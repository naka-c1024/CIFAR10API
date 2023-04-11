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

class Classifier:
    def __init__(self, model_path):
        self.net = Net()
        self.net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.net.eval()

        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

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

@app.route("/")
def index():
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
