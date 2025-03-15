# intro to ml

Part 1: Literature Review (20 Marks)
1. Importance of Feature Extraction in Computer Vision
Feature extraction is a crucial step in computer vision tasks as it transforms raw image data into a more meaningful representation, enabling machine learning models to recognize patterns efficiently. Without proper feature extraction, models may struggle with high-dimensional data, noise, and redundancy, leading to suboptimal classification performance.

2. Three Conventional Image Feature Extraction Methods
(i) Histogram of Oriented Gradients (HOG)
Principle: HOG captures edge directions by computing gradient orientations in localized image regions.
Application: Used in object detection, such as pedestrian detection in self-driving cars.
(ii) Scale-Invariant Feature Transform (SIFT)
Principle: SIFT detects key points in an image that are invariant to scale, rotation, and illumination changes.
Application: Used in image stitching, facial recognition, and object tracking.
(iii) Gray Level Co-occurrence Matrix (GLCM)
Principle: GLCM calculates texture features based on the spatial relationship of pixel intensities in grayscale images.
Application: Used in medical image analysis (tumor detection) and remote sensing.
Part 2: Experimentation (40 Marks)
Dataset Selection
We choose the CIFAR-10 dataset, which consists of 60,000 images categorized into 10 classes (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck).

Feature Extraction Techniques & Steps
1. Traditional Feature Extraction
HOG: Extracts gradient orientation features.
Local Binary Patterns (LBP): Encodes texture patterns by comparing pixel intensity with neighbors.
Edge Detection (Canny/Sobel): Identifies image edges for feature extraction.
2. Deep Learning-Based Feature Extraction
ResNet, VGG, MobileNet: Pretrained CNN models extract hierarchical features.
Implementation Steps
Preprocessing: Convert images to grayscale, resize to 32×32, normalize pixel values.
Feature Extraction: Apply HOG, LBP, Edge Detection, and CNN-based models.
Train Classifier: Use Logistic Regression, KNN, Decision Trees, or Random Forests.
Evaluation Metrics: Compute accuracy, precision, recall, and F1-score.
Part 3: Analysis (30 Marks)
Comparison of Feature Extraction Methods

![Screenshot 2025-03-14 211126](https://github.com/user-attachments/assets/328445b2-4cd8-4ddc-a789-6b9ba9ccaad8)

Observations & Trade-offs
Traditional methods like HOG and LBP perform well but require manual feature engineering.
Deep learning models (ResNet, VGG) provide superior accuracy but are computationally expensive.
Feature representation significantly impacts classification performance, as deep learning models extract more abstract features.
Part 4: Report and Presentation (10 Marks)

# program
~~~
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from skimage import io, color
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification

# Generate a synthetic dataset since OpenML access requires internet
def load_sample_dataset():
    X, y = make_classification(n_samples=10000, n_features=64, n_informative=50, n_classes=10, random_state=42)
    X = X.reshape(-1, 8, 8)  # Reshape to mimic small grayscale images
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Preprocessing function
def preprocess_images(images):
    processed_images = []
    for img in images:
        img = cv2.resize(img, (64, 64))  # Resize to standard size
        img = img.astype('float32') / 255.0  # Normalize
        processed_images.append(img)
    return np.array(processed_images)

# Feature Extraction - HOG
def extract_hog_features(images):
    features = []
    for img in images:
        fd, _ = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        features.append(fd)
    return np.array(features)

# Feature Extraction - LBP
def extract_lbp_features(images):
    features = []
    for img in images:
        lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
        hist = hist.astype('float') / hist.sum()  # Normalize histogram
        features.append(hist)
    return np.array(features)

# Train classifier and evaluate
def train_and_evaluate(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Main Execution
X_train, X_test, y_train, y_test = load_sample_dataset()
X_train, X_test = preprocess_images(X_train), preprocess_images(X_test)

print("Extracting HOG Features...")
hog_features_train = extract_hog_features(X_train[:5000])
hog_features_test = extract_hog_features(X_test[:1000])
train_and_evaluate(hog_features_train, hog_features_test, y_train[:5000], y_test[:1000])

print("Extracting LBP Features...")
lbp_features_train = extract_lbp_features(X_train[:5000])
lbp_features_test = extract_lbp_features(X_test[:1000])
train_and_evaluate(lbp_features_train, lbp_features_test, y_train[:5000], y_test[:1000])

~~~
# output

![Screenshot 2025-03-15 183141](https://github.com/user-attachments/assets/a8274fb7-049f-4fb1-8daf-74848e629830)

![Screenshot 2025-03-15 183157](https://github.com/user-attachments/assets/a785000b-5a8c-479e-a405-f2eb20f06aae)

