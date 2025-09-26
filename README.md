🛡️ Phishing Website Detection Using Random Forest

This project uses Machine Learning to classify websites as phishing or legitimate using the UCI Phishing Website Dataset. It demonstrates loading .arff data, preprocessing, training a Random Forest model, and evaluating its performance.

📂 Project Overview
Dataset Source: https://archive.ics.uci.edu/ml/datasets/Phishing+Websites
Goal: Predict whether a website is phishing (0) or legitimate (1).
Model Used: RandomForestClassifier from scikit-learn

📑 Features of the Project

> Loads .arff dataset from the UCI repository using requests and scipy.io.arff.
> Converts target column values from -1 → 0 (phishing) and 1 → 1 (legitimate).
> Splits data into train and test sets.
> Trains a Random Forest Classifier.
> Evaluates the model using accuracy, confusion matrix, and classification report.
> Tests a single example from the test set.

🗂️ Project Structure
phishing-detection/
│
├── phishing_random_forest.py   # Main Python script
└── README.md                   # Project documentation

⚙️ Requirements
Install dependencies via pip: pip install pandas scikit-learn scipy requests

▶️ How to Run the Code

Clone or Download this repository.
Open the Python script in your IDE or run it in Google Colab / Jupyter Notebook.
Run the script:
python phishing_random_forest.py

The script will:
Download the dataset
Preprocess data
Train and evaluate the model
Print model performance metrics

📊 Model Evaluation Metrics

Accuracy Score
Confusion Matrix
Classification Report (Precision, Recall, F1-score)

🧪 Single Example Prediction
The script tests the model on a single example from the test set and shows:
Actual label (Phishing / Legitimate)
Predicted label
Prediction probabilities

🔮 Future Improvements
Add feature importance visualization.
Experiment with other models (Logistic Regression, Gradient Boosting).
Perform hyperparameter tuning (GridSearchCV).


