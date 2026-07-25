# Weather-Prediction-from-Travel-Images

This repository contains **Assignment 3** for the **Machine Learning and Data Science (ENCS5341)** course.

The project predicts weather conditions from outdoor travel images using traditional machine-learning models and transfer learning.

## Team Members

- Noureddin Etkaidek — 1220162
- Ibraheem Sleet — 1220200
- Mohammed Yousef — 1220041

**Instructor:** Dr. Yazan Abu Farha

## Weather Classes

The system classifies images into five categories:

- Sunny
- Cloudy
- Rainy
- Snowy
- Not Clear

## Project Pipeline

1. Collect CSV files from student submission folders.
2. Standardise weather labels.
3. Download images from their URLs.
4. Extract colour, texture, brightness, and sky-region features.
5. Train and compare multiple classifiers.
6. Evaluate the best model using accuracy, classification reports, and a confusion matrix.
7. Save the trained model for future predictions.

## Models

The project compares:

- K-Nearest Neighbours
- Random Forest
- Gradient Boosting
- Support Vector Machine
- Multilayer Perceptron
- ResNet50 transfer learning when TensorFlow is available

The best reported test accuracy was **74.31%**, achieved by both the tuned SVM and ResNet50 models.

## Repository Structure

```text
.
├── weather_prediction.py
├── Report.pdf
├── dataset/
│   ├── student_1/
│   │   └── data.csv
│   └── student_2/
│       └── data.csv
└── README.md
```

Each CSV file must contain at least these columns:

```text
Image URL
Weather
```

## Installation

```bash
pip install pandas numpy requests pillow tqdm scikit-learn matplotlib seaborn
```

TensorFlow is optional and is required only for ResNet50 training:

```bash
pip install tensorflow
```

## Run

Place the dataset folders inside `dataset/`, then run:

```bash
python weather_prediction.py
```

## Generated Files

The script may generate:

```text
downloaded_images/
weather_model.pkl
training_results.png
weather_distribution.png
failed_urls.txt
```

## Predict a New Image

After training:

```python
from weather_prediction import predict_weather

weather, confidence = predict_weather("image.jpg")

print("Weather:", weather)
print("Confidence:", confidence)
```

## Main Challenges

- Strong class imbalance, especially for Rainy and Snowy images
- Visual similarity between Sunny, Cloudy, and Not Clear conditions
- Broken or restricted image URLs
- Variation in image size, lighting, geography, and season
