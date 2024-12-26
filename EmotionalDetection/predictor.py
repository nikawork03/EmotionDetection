import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned data
file_path = 'emotionalFiles/cleaned_sentiment_data.csv'
data = pd.read_csv(file_path)

# Text Vectorization using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(data['sentence']).toarray()

# Target labels
y = data['emotion']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model 1: Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)

print("Naive Bayes Performance:")
print("Accuracy:", accuracy_score(y_test, nb_predictions))
print("Classification Report:\n", classification_report(y_test, nb_predictions))

# Save Naive Bayes Predictions
nb_results = pd.DataFrame({
    'sentence': data.iloc[y_test.index]['sentence'],
    'predicted_emotion': nb_predictions,
    'actual_emotion': y_test.values
})
nb_correct = nb_results[nb_results['predicted_emotion'] == nb_results['actual_emotion']]
nb_incorrect = nb_results[nb_results['predicted_emotion'] != nb_results['actual_emotion']]
nb_correct.to_csv('predictions/Naive_Bayes_correct_predictions.csv', index=False)
nb_incorrect.to_csv('predictions/Naive_Bayes_incorrect_predictions.csv', index=False)

# Confusion Matrix for Naive Bayes
plt.figure(figsize=(8, 6))
nb_cm = confusion_matrix(y_test, nb_predictions, labels=y.unique())
sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Blues', xticklabels=y.unique(), yticklabels=y.unique())
plt.title("Confusion Matrix - Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Model 2: Support Vector Machine (SVM)
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

print("\nSVM Performance:")
print("Accuracy:", accuracy_score(y_test, svm_predictions))
print("Classification Report:\n", classification_report(y_test, svm_predictions))

# Save SVM Predictions
svm_results = pd.DataFrame({
    'sentence': data.iloc[y_test.index]['sentence'],
    'predicted_emotion': svm_predictions,
    'actual_emotion': y_test.values
})
svm_correct = svm_results[svm_results['predicted_emotion'] == svm_results['actual_emotion']]
svm_incorrect = svm_results[svm_results['predicted_emotion'] != svm_results['actual_emotion']]
svm_correct.to_csv('predictions/SVM_correct_predictions.csv', index=False)
svm_incorrect.to_csv('predictions/SVM_incorrect_predictions.csv', index=False)

# Confusion Matrix for SVM
plt.figure(figsize=(8, 6))
svm_cm = confusion_matrix(y_test, svm_predictions, labels=y.unique())
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Greens', xticklabels=y.unique(), yticklabels=y.unique())
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Comparison of Model Performances
nb_accuracy = accuracy_score(y_test, nb_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)

print("\nModel Comparison:")
print(f"Naive Bayes Accuracy: {nb_accuracy * 100:.2f}%")
print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")