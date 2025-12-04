## 🛡️ Phishing Website Detection using Machine Learning

A machine learning project that detects phishing websites using the UCI Phishing Websites Dataset.
This project loads the dataset from the UCI Repository (.arff format), preprocesses it, trains a Random Forest Classifier, evaluates it, and performs a test prediction.

<div align="center">








</div>

## 🚀 Features
-	Automatically downloads the Phishing Websites Dataset (UCI ML Repository)
-	Handles ARFF format and converts it into a pandas DataFrame
-	Cleans and maps labels:
    -	-1 → 0 (Phishing)
    -	1 → 1 (Legitimate)
-	Splits dataset into Training (80%) and Testing (20%)
-	Trains a RandomForestClassifier with 100 trees
-	Evaluates model with:
    -	Accuracy
    -	Confusion Matrix
    -	Detailed Classification Report (Precision, Recall, F1-score)
-	Performs a single sample prediction test
-	Simple, clean, and easy to extend

## 📂 Project Structure
```
Phishing-Websites-Detection/
│
├── project_phishing_websites_dataset.py     # Main ML script
├── README.md                                 # Project documentation
└── (output generated on runtime)
```

## 📦 Dataset Used
UCI Phishing Websites Dataset
Link: https://archive.ics.uci.edu/ml/datasets/phishing+websites
-	Format: .arff
-	Total Features: 30
-	Target Column: Result
    -	0 → Phishing
    -	1 → Legitimate
Data is automatically fetched via:
```
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
```

## 🛠️ Installation & Requirements
Install dependencies:
```
pip install pandas scikit-learn scipy requests
```

## ▶️ How to Run
Run the main script:
```
python project_phishing_websites_dataset.py
```
The script will:
1.	Download the dataset
2.	Convert ARFF → DataFrame
3.	Preprocess
4.	Train Random Forest
5.	Evaluate
6.	Run a sample prediction

## 🤖 Model Used
RandomForestClassifier
-	100 decision trees
-	random_state=42
-	n_jobs=-1 (full CPU usage)
Why Random Forest?
-	Excellent for tabular data
-	Handles mixed-feature datasets well
-	Great accuracy with minimal tuning
-	Resistant to overfitting

📊 Model Performance (Sample Output)
The script prints:

✔ Accuracy (example)
```Model Accuracy: 0.9550 (95.50%)```
✔ Confusion Matrix (example)
```
[[TN  FP]
 [FN  TP] ]
```
✔ Classification Report
Shows:
- Precision
- Recall
- F1-score
- Support

## 🔍 Single Test Prediction
The script predicts one example from the test set:
```
Actual Label: Legitimate
Predicted Label: Legitimate
Prediction Probabilities: [Phishing_prob  Legitimate_prob]
```
This helps verify real-case model behavior.

## 🧠 How It Works (Summary)

1. Load ARFF file from URL
2. Convert to pandas DataFrame
3. Map labels (-1 → 0, 1 → 1)
4. Train-test split
5. Train Random Forest
6. Predict + evaluate
7. Display metrics

## 🧩 Future Improvements

- Add feature importance visualization
- Try advanced models (XGBoost, LightGBM)
- Implement feature scaling
- Build a Flask-based web API for predictions
- Add hyperparameter tuning (GridSearchCV)

## 📘 About
A machine learning project demonstrating phishing website detection using traditional ML and the UCI dataset.

