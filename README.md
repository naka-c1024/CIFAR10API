# 画像分類アプリ

CIFAR10を学習した画像分類モデルで「飛行機, 自動車, 鳥, 猫, 鹿, 犬, 蛙, 馬, 船, トラック」を分類するwebアプリケーションです。

### プロダクトURL: https://cifar10api-sg3vqkre3q-an.a.run.app/

# Features
- サイト上で画像分類
- 分類APIを実装

# DEMO

https://user-images.githubusercontent.com/78196739/231756704-1e816820-9859-4838-83b0-cc4bf4d391d0.mp4

# APIの使用方法

## URI

\<URI>の部分を変更してください。

```bash
curl -X POST -H 'Content-Type: application/json' -d '{"image_uri": "<URI>"}' https://cifar10api-sg3vqkre3q-an.a.run.app/predict
```

## File

\<File Path>の部分を変更してください。

```
curl -X POST -F image_file=@<File Path> https://cifar10api-sg3vqkre3q-an.a.run.app/predict
```

## 出力例
```
{
	"predictions": [
		{
			"classification_results": [
				"airplane"
			],
			"score": [
				0.9969457983970642
			]
		}
	]
}
```

# Usage

## Git clone

```
git clone https://github.com/naka-c1024/CIFAR10API.git
```

## Preparation

### Dockerを使用する場合

```bash
docker build -t cifar10_api .
docker run -v $(pwd):/app -p 8080:8080 -it cifar10_api bash
```

### Dockerを使用せずローカル環境で実行したい場合

#### Requirement

- Flask 2.1.0
- torch 2.0.0
- torchvision 0.15.1

#### Installation

```
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

# 使用技術

### 環境: Docker

### フロントエンド: HTML，CSS (TailWindCSS)

### バックエンド: Python (Flask)

### DNN: PyTorch

### デプロイ: Google Cloud Platform (Cloud Run)
