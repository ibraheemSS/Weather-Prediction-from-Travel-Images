"""
Weather Prediction from Travel Images
ENCS5341 - Assignment 3

This script uses traditional ML with color/texture features for weather classification.
Compatible with Python 3.13 (no PyTorch/TensorFlow required)

Pipeline:
1. Loads all CSV files from student submissions
2. Downloads images from URLs (handles broken URLs)
3. Extracts visual features (color histograms, texture, etc.)
4. Trains ML classifiers to predict weather conditions
"""

import os
import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import warnings
from tqdm import tqdm
import pickle
from collections import Counter
import json

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Try importing TensorFlow for CNN

try:
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import (
        preprocess_input as resnet_preprocess,
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

warnings.filterwarnings("ignore")

# Configuration
DATASET_DIR = "dataset"
IMAGES_DIR = "downloaded_images"
MODEL_SAVE_PATH = "weather_model.pkl"
IMAGE_SIZE = (128, 128)  # Smaller size for feature extraction
RANDOM_STATE = 42

# Weather categories to predict
WEATHER_CLASSES = ["Sunny", "Rainy", "Cloudy", "Snowy", "Not Clear"]


def collect_csv_files(dataset_dir):
    """Collect all CSV files from student submission folders."""
    all_data = []
    csv_files_found = 0

    print("=" * 60)
    print("Phase 1: Collecting CSV files from student submissions")
    print("=" * 60)

    for student_folder in sorted(os.listdir(dataset_dir)):
        student_path = os.path.join(dataset_dir, student_folder)

        if os.path.isdir(student_path):
            for file in os.listdir(student_path):
                if file.endswith(".csv"):
                    csv_path = os.path.join(student_path, file)
                    try:
                        # Try different encodings
                        df = None
                        for encoding in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
                            try:
                                df = pd.read_csv(csv_path, encoding=encoding)
                                break
                            except UnicodeDecodeError:
                                continue

                        if df is None:
                            continue

                        # Standardize column names
                        df.columns = df.columns.str.strip()

                        # Check if required columns exist
                        if "Image URL" in df.columns and "Weather" in df.columns:
                            df["Student_ID"] = student_folder
                            df["Source_File"] = file
                            all_data.append(df)
                            csv_files_found += 1
                        else:
                            print(f"  Warning: Missing columns in {csv_path}")

                    except Exception as e:
                        print(f"  Error reading {csv_path}: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n✓ Found {csv_files_found} CSV files")
        print(f"✓ Total images: {len(combined_df)}")
        return combined_df
    else:
        print("No valid CSV files found!")
        return pd.DataFrame()


def clean_weather_labels(df):
    """Standardize weather labels."""
    print("\n" + "=" * 60)
    print("Phase 2: Cleaning and standardizing weather labels")
    print("=" * 60)

    # Create a copy
    df = df.copy()

    # Standardize weather column
    df["Weather_Original"] = df["Weather"]
    df["Weather"] = df["Weather"].astype(str).str.strip().str.lower()

    # Mapping variations to standard labels
    weather_mapping = {
        "sunny": "Sunny",
        "clear": "Sunny",
        "bright": "Sunny",
        "rainy": "Rainy",
        "rain": "Rainy",
        "raining": "Rainy",
        "cloudy": "Cloudy",
        "clouds": "Cloudy",
        "overcast": "Cloudy",
        "partly cloudy": "Cloudy",
        "snowy": "Snowy",
        "snow": "Snowy",
        "winter": "Snowy",
        "not clear": "Not Clear",
        "unclear": "Not Clear",
        "night": "Not Clear",
        "not clear (night lighting)": "Not Clear",
        "foggy": "Cloudy",
        "hazy": "Cloudy",
    }

    def map_weather(weather):
        weather_lower = str(weather).lower().strip()
        for key, value in weather_mapping.items():
            if key in weather_lower:
                return value
        return "Not Clear"  # Default for unknown

    df["Weather_Clean"] = df["Weather"].apply(map_weather)

    # Print distribution
    print("\nWeather distribution:")
    weather_counts = df["Weather_Clean"].value_counts()
    for weather, count in weather_counts.items():
        print(f"  {weather}: {count}")

    return df


def download_image(url, timeout=10):
    """Download image from URL with error handling."""
    try:
        # Handle different URL formats
        url = str(url).strip()

        # Skip if empty or NaN
        if not url or url == "nan" or pd.isna(url):
            return None

        # Add https if missing
        if not url.startswith("http"):
            url = "https://" + url

        # Set headers to mimic browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        }

        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        # Check content length - skip very large files
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > 20 * 1024 * 1024:  # > 20MB
            return None

        img = Image.open(BytesIO(response.content))
        img = img.convert("RGB")
        return img

    except Exception as e:
        return None


def download_all_images(df, images_dir):
    """Download all images and save them locally."""
    print("\n" + "=" * 60)
    print("Phase 3: Downloading images from URLs")
    print("=" * 60)

    os.makedirs(images_dir, exist_ok=True)

    successful = 0
    failed = 0
    failed_urls = []
    image_paths = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        url = row["Image URL"]
        weather = row["Weather_Clean"]

        # Create filename
        safe_filename = f"{idx}_{weather.replace(' ', '_')}.jpg"
        save_path = os.path.join(images_dir, safe_filename)

        # Check if already downloaded
        if os.path.exists(save_path):
            # Verify the image is valid
            try:
                img = Image.open(save_path)
                img.verify()
                image_paths.append(save_path)
                valid_indices.append(idx)
                successful += 1
                continue
            except:
                os.remove(save_path)

        # Download image
        img = download_image(url)

        if img is not None:
            try:
                img.save(save_path, "JPEG", quality=95)
                image_paths.append(save_path)
                valid_indices.append(idx)
                successful += 1
            except Exception as e:
                failed += 1
                failed_urls.append(url)
        else:
            failed += 1
            failed_urls.append(url)

    print(f"\n✓ Successfully downloaded: {successful}")
    print(f"✗ Failed downloads: {failed}")
    print(f"  Success rate: {successful/(successful+failed)*100:.1f}%")

    # Save failed URLs for reference
    if failed_urls:
        with open("failed_urls.txt", "w") as f:
            for url in failed_urls:
                f.write(f"{url}\n")
        print(f"  Failed URLs saved to: failed_urls.txt")

    return image_paths, valid_indices


def extract_color_features(img):
    """Extract color histogram features from an image."""
    # Convert to numpy array
    img_array = np.array(img)

    features = []

    # Color histograms for each channel (R, G, B)
    for channel in range(3):
        hist, _ = np.histogram(img_array[:, :, channel], bins=32, range=(0, 256))
        hist = hist.astype(float) / hist.sum()  # Normalize
        features.extend(hist)

    # Mean and std of each channel
    for channel in range(3):
        features.append(np.mean(img_array[:, :, channel]))
        features.append(np.std(img_array[:, :, channel]))

    # Brightness (average of all pixels)
    features.append(np.mean(img_array))

    # Color ratios
    r_mean = np.mean(img_array[:, :, 0])
    g_mean = np.mean(img_array[:, :, 1])
    b_mean = np.mean(img_array[:, :, 2])
    total = r_mean + g_mean + b_mean + 1e-6

    features.append(r_mean / total)
    features.append(g_mean / total)
    features.append(b_mean / total)

    return features


def extract_texture_features(img):
    """Extract simple texture features using gradients."""
    # Convert to grayscale
    gray = img.convert("L")
    gray_array = np.array(gray, dtype=np.float32)

    features = []

    # Gradient features (edge information)
    grad_x = np.diff(gray_array, axis=1)
    grad_y = np.diff(gray_array, axis=0)

    features.append(np.mean(np.abs(grad_x)))
    features.append(np.std(np.abs(grad_x)))
    features.append(np.mean(np.abs(grad_y)))
    features.append(np.std(np.abs(grad_y)))

    # Gradient magnitude
    min_h = min(grad_x.shape[0], grad_y.shape[0])
    min_w = min(grad_x.shape[1], grad_y.shape[1])
    grad_mag = np.sqrt(grad_x[:min_h, :min_w] ** 2 + grad_y[:min_h, :min_w] ** 2)

    features.append(np.mean(grad_mag))
    features.append(np.std(grad_mag))

    # Contrast (std of grayscale)
    features.append(np.std(gray_array))

    # Entropy approximation
    hist, _ = np.histogram(gray_array, bins=64, range=(0, 256))
    hist = hist.astype(float) / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    features.append(entropy)

    return features


def extract_sky_features(img):
    """Extract features from the upper portion (likely sky)."""
    img_array = np.array(img)

    # Get top 30% of image (sky region)
    h = img_array.shape[0]
    sky_region = img_array[: int(h * 0.3), :, :]

    features = []

    # Sky color statistics
    for channel in range(3):
        features.append(np.mean(sky_region[:, :, channel]))
        features.append(np.std(sky_region[:, :, channel]))

    # Blue ratio in sky (higher for sunny days)
    r_mean = np.mean(sky_region[:, :, 0])
    g_mean = np.mean(sky_region[:, :, 1])
    b_mean = np.mean(sky_region[:, :, 2])

    # Blue dominance ratio
    total = r_mean + g_mean + b_mean + 1e-6
    features.append(b_mean / total)
    features.append((b_mean - r_mean) / 255.0)  # Blue vs Red difference

    # Brightness of sky
    features.append(np.mean(sky_region))

    # Variance (cloudy skies have lower variance)
    features.append(np.var(sky_region))

    return features


def extract_all_features(image_path, image_size=(128, 128)):
    """Extract all features from an image."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(image_size, Image.Resampling.LANCZOS)

        features = []

        # Color features
        color_features = extract_color_features(img)
        features.extend(color_features)

        # Texture features
        texture_features = extract_texture_features(img)
        features.extend(texture_features)

        # Sky features
        sky_features = extract_sky_features(img)
        features.extend(sky_features)

        return np.array(features)

    except Exception as e:
        return None


def prepare_features(image_paths, labels):
    """Extract features from all images."""
    print("\n" + "=" * 60)
    print("Phase 4: Extracting visual features")
    print("=" * 60)

    X = []
    y = []
    valid_paths = []

    for img_path, label in tqdm(
        zip(image_paths, labels), total=len(image_paths), desc="Extracting features"
    ):
        features = extract_all_features(img_path, IMAGE_SIZE)

        if features is not None:
            X.append(features)
            y.append(label)
            valid_paths.append(img_path)

    X = np.array(X)
    y = np.array(y)

    print(f"\n✓ Features extracted from {len(X)} images")
    print(f"  Feature vector size: {X.shape[1]}")

    return X, y, valid_paths


def load_images_for_cnn(image_paths, target_size=(224, 224)):
    """Load and preprocess images for CNN (ResNet50 input format)."""
    images = []
    valid_paths = []

    for img_path in tqdm(image_paths, desc="Loading images for CNN"):
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.array(img)
            images.append(img_array)
            valid_paths.append(img_path)
        except Exception as e:
            continue

    return np.array(images), valid_paths


def create_cnn_model(num_classes):
    """Create CNN model using pre-trained ResNet50 with transfer learning."""
    # Load pre-trained ResNet50 (without top classification layer)
    base_model = ResNet50(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )

    # Freeze base model layers (transfer learning)
    base_model.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    # Create model
    model = Model(inputs=base_model.input, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_cnn(image_paths, y, label_encoder):
    """Train CNN model using transfer learning with ResNet50."""
    if not TF_AVAILABLE:
        print("\n⚠ TensorFlow not available - Skipping CNN training")
        print("  Install with: pip install tensorflow")
        return None, None, None

    print("\n" + "=" * 60)
    print("Training CNN with Transfer Learning (ResNet50)")
    print("=" * 60)

    # Load images for CNN
    print("\nLoading images for CNN...")
    X_images, valid_paths = load_images_for_cnn(image_paths)
    y_valid = y[: len(X_images)]

    # Encode labels
    y_encoded = label_encoder.transform(y_valid)

    # Split data
    X_train_img, X_test_img, y_train_cnn, y_test_cnn = train_test_split(
        X_images,
        y_encoded,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    print(f"Training samples: {len(X_train_img)}")
    print(f"Test samples: {len(X_test_img)}")

    # Preprocess for ResNet50
    X_train_processed = resnet_preprocess(X_train_img.astype("float32"))
    X_test_processed = resnet_preprocess(X_test_img.astype("float32"))

    # Create model
    num_classes = len(label_encoder.classes_)
    model = create_cnn_model(num_classes)

    print(f"\nModel Architecture:")
    print(f"  Base: ResNet50 (pre-trained on ImageNet)")
    print(f"  Custom layers: GAP → Dense(512) → Dense(256) → Dense({num_classes})")
    print(f"  Total trainable parameters: {model.count_params():,}")

    # Callbacks
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )

    # Train model
    print("\nTraining CNN...")
    history = model.fit(
        X_train_processed,
        y_train_cnn,
        validation_data=(X_test_processed, y_test_cnn),
        epochs=20,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    # Evaluate
    y_pred_cnn = np.argmax(model.predict(X_test_processed, verbose=0), axis=1)
    acc = accuracy_score(y_test_cnn, y_pred_cnn)

    print(f"\nCNN (ResNet50 Transfer Learning):")
    print(f"  Test Accuracy: {acc*100:.2f}%")
    print(f"  Training epochs: {len(history.history['loss'])}")

    return model, acc, y_pred_cnn


def train_models(X, y, label_encoder):
    """Train multiple classifiers and select the best one."""
    print("\n" + "=" * 60)
    print("Phase 5: Training ML classifiers")
    print("=" * 60)

    # Encode labels
    y_encoded = label_encoder.transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define classifiers
    classifiers = {
        "KNN (k=1)": KNeighborsClassifier(
            n_neighbors=1, weights="uniform", metric="euclidean"
        ),
        "KNN (k=3)": KNeighborsClassifier(
            n_neighbors=3, weights="distance", metric="euclidean"
        ),
        "KNN (k=5)": KNeighborsClassifier(
            n_neighbors=5, weights="distance", metric="euclidean"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE
        ),
        "SVM": SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            random_state=RANDOM_STATE,
            probability=True,
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=500,
            random_state=RANDOM_STATE,
            early_stopping=True,
        ),
    }

    results = {}
    best_acc = 0
    best_model = None
    best_model_name = None

    print("\nTraining and evaluating classifiers...")
    print("-" * 50)

    for name, clf in classifiers.items():
        print(f"\n{name}:")

        # Train
        if name in ["SVM", "Neural Network", "KNN (k=1)", "KNN (k=3)", "KNN (k=5)"]:
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        results[name] = {"accuracy": acc, "predictions": y_pred}

        print(f"  Accuracy: {acc*100:.2f}%")

        # Cross-validation
        if name in ["SVM", "Neural Network", "KNN (k=1)", "KNN (k=3)", "KNN (k=5)"]:
            cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
        else:
            cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
        print(
            f"  CV Score: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)"
        )

        if acc > best_acc:
            best_acc = acc
            best_model = clf
            best_model_name = name

    print("\n" + "=" * 50)
    print(f"Best Model: {best_model_name} (Accuracy: {best_acc*100:.2f}%)")

    return best_model, best_model_name, scaler, X_test, y_test, X_test_scaled, results


def evaluate_best_model(
    model, model_name, X_test, y_test, label_encoder, use_scaled=False
):
    """Detailed evaluation of the best model."""
    print("\n" + "=" * 60)
    print("Phase 6: Detailed Model Evaluation")
    print("=" * 60)

    y_pred = model.predict(X_test)

    # Get unique classes present in test and predictions
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    target_names = [label_encoder.classes_[i] for i in unique_labels]

    print(f"\nClassification Report for {model_name}:")
    print("-" * 50)
    print(
        classification_report(
            y_test,
            y_pred,
            labels=unique_labels,
            target_names=target_names,
            zero_division=0,
        )
    )

    return y_pred


def plot_results(y_test, y_pred, label_encoder, results):
    """Plot confusion matrix and model comparison."""
    print("\n" + "=" * 60)
    print("Phase 7: Generating visualizations")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Get unique classes present in test and predictions
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    class_names = [label_encoder.classes_[i] for i in unique_labels]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix")

    # Model Comparison
    model_names = list(results.keys())
    accuracies = [results[name]["accuracy"] * 100 for name in model_names]

    bars = axes[1].barh(
        model_names, accuracies, color=["steelblue", "coral", "green", "purple"]
    )
    axes[1].set_xlabel("Accuracy (%)")
    axes[1].set_title("Model Comparison")
    axes[1].set_xlim(0, 100)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        axes[1].text(
            acc + 1, bar.get_y() + bar.get_height() / 2, f"{acc:.1f}%", va="center"
        )

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("✓ Results saved to: training_results.png")


def plot_weather_distribution(df):
    """Plot the distribution of weather classes."""
    fig, ax = plt.subplots(figsize=(10, 5))

    weather_counts = df["Weather_Clean"].value_counts()

    colors = ["gold", "gray", "lightblue", "white", "lavender"]
    bars = ax.bar(
        weather_counts.index, weather_counts.values, color=colors, edgecolor="black"
    )

    ax.set_xlabel("Weather Category")
    ax.set_ylabel("Number of Images")
    ax.set_title("Distribution of Weather Classes in Dataset")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("weather_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("✓ Distribution saved to: weather_distribution.png")


def save_model(model, scaler, label_encoder, model_name):
    """Save the trained model and preprocessing objects."""
    model_data = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "model_name": model_name,
        "weather_classes": WEATHER_CLASSES,
        "image_size": IMAGE_SIZE,
    }

    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(model_data, f)

    print(f"✓ Model saved to: {MODEL_SAVE_PATH}")


def predict_weather(image_path, model_path=MODEL_SAVE_PATH):
    """Predict weather for a single image."""
    # Load model
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    scaler = model_data["scaler"]
    label_encoder = model_data["label_encoder"]
    model_name = model_data["model_name"]

    # Extract features
    features = extract_all_features(image_path, IMAGE_SIZE)

    if features is None:
        return None, 0

    features = features.reshape(1, -1)

    # Scale if needed
    if model_name in ["SVM", "Neural Network"]:
        features = scaler.transform(features)

    # Predict
    prediction = model.predict(features)[0]
    weather = label_encoder.inverse_transform([prediction])[0]

    # Get probability if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        confidence = proba[prediction] * 100
    else:
        confidence = 0

    return weather, confidence


def main():
    """Main function to run the entire pipeline."""
    print("\n" + "=" * 60)
    print("   WEATHER PREDICTION FROM TRAVEL IMAGES")
    print("   ENCS5341 - Assignment 3")
    print("=" * 60)

    # Step 1: Collect CSV files
    df = collect_csv_files(DATASET_DIR)

    if df.empty:
        print("No data found. Exiting.")
        return

    # Step 2: Clean weather labels
    df = clean_weather_labels(df)

    # Plot weather distribution
    plot_weather_distribution(df)

    # Step 3: Download images
    image_paths, valid_indices = download_all_images(df, IMAGES_DIR)

    if len(image_paths) < 10:
        print("Not enough images downloaded. Exiting.")
        return

    # Get valid labels
    labels = df.loc[valid_indices, "Weather_Clean"].tolist()

    # Step 4: Extract features
    X, y, valid_paths = prepare_features(image_paths, labels)

    # Step 5: Prepare label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(WEATHER_CLASSES)

    # Step 6: Train models
    best_model, best_model_name, scaler, X_test, y_test, X_test_scaled, results = (
        train_models(X, y, label_encoder)
    )

    # Step 6.5: Train CNN (if TensorFlow is available)
    if TF_AVAILABLE:
        cnn_model, cnn_acc, cnn_pred = train_cnn(valid_paths, y, label_encoder)
        if cnn_model is not None:
            results["CNN (ResNet50)"] = {"accuracy": cnn_acc, "predictions": cnn_pred}
            # Update best model if CNN is better
            if cnn_acc > results[best_model_name]["accuracy"]:
                print(f"\n✓ CNN outperforms {best_model_name}!")
                print(f"  New best accuracy: {cnn_acc*100:.2f}%")
    else:
        print("\n" + "=" * 60)
        print("⚠ CNN Training Skipped - TensorFlow Not Available")
        print("=" * 60)
        print("\nTo include CNN in the comparison:")
        print("  pip install tensorflow")
        print("\nCNN uses transfer learning with pre-trained ResNet50,")
        print("which typically achieves higher accuracy than traditional ML.")

    # Step 7: Evaluate best model
    if best_model_name in ["SVM", "Neural Network"]:
        y_pred = evaluate_best_model(
            best_model, best_model_name, X_test_scaled, y_test, label_encoder
        )
    else:
        y_pred = evaluate_best_model(
            best_model, best_model_name, X_test, y_test, label_encoder
        )

    # Step 8: Plot results
    plot_results(y_test, y_pred, label_encoder, results)

    # Step 9: Save model
    save_model(best_model, scaler, label_encoder, best_model_name)

    print("\n" + "=" * 60)
    print("   TRAINING COMPLETE!")
    print("=" * 60)
    print("\nSaved files:")
    print(f"  - Model: {MODEL_SAVE_PATH}")
    print(f"  - Results Plot: training_results.png")
    print(f"  - Distribution Plot: weather_distribution.png")
    print("\nTo predict weather for a new image:")
    print("  from weather_prediction import predict_weather")
    print("  weather, confidence = predict_weather('image.jpg')")


if __name__ == "__main__":
    main()
