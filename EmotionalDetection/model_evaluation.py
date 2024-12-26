import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt

# Load the cleaned data
file_path = 'emotionalFiles/cleaned_sentiment_data.csv'
data = pd.read_csv(file_path)

# Text Vectorization using TF-IDF
max_features_list = [1000, 2000, 3000, 5000]
results = []

for max_features in max_features_list:
    print(f"Running for max_features = {max_features}...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    X = vectorizer.fit_transform(data['sentence']).toarray()

    # Target labels
    y = data['emotion']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Model 1: Naive Bayes with Cross-Validation
    nb_model = MultinomialNB()
    nb_cv_scores = cross_val_score(nb_model, X, y, cv=5, scoring='accuracy')
    nb_model.fit(X_train, y_train)
    nb_predictions = nb_model.predict(X_test)

    # Model 2: SVM with Cross-Validation
    svm_model = SVC(kernel='linear', random_state=42)
    svm_cv_scores = cross_val_score(svm_model, X, y, cv=5, scoring='accuracy')
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)

    # Save results
    results.append({
        'max_features': max_features,
        'nb_cv_mean': nb_cv_scores.mean(),
        'nb_cv_std': nb_cv_scores.std(),
        'svm_cv_mean': svm_cv_scores.mean(),
        'svm_cv_std': svm_cv_scores.std()
    })

    print(f"Naive Bayes CV Accuracy: {nb_cv_scores.mean() * 100:.2f}% \u00b1 {nb_cv_scores.std() * 100:.2f}%")
    print(f"SVM CV Accuracy: {svm_cv_scores.mean() * 100:.2f}% \u00b1 {svm_cv_scores.std() * 100:.2f}%")

# Convert results to DataFrame for visualization
results_df = pd.DataFrame(results)
plt.figure(figsize=(10, 6))
plt.plot(results_df['max_features'], results_df['nb_cv_mean'], label='Naive Bayes', marker='o')
plt.fill_between(results_df['max_features'],
                 results_df['nb_cv_mean'] - results_df['nb_cv_std'],
                 results_df['nb_cv_mean'] + results_df['nb_cv_std'], alpha=0.2)
plt.plot(results_df['max_features'], results_df['svm_cv_mean'], label='SVM', marker='o')
plt.fill_between(results_df['max_features'],
                 results_df['svm_cv_mean'] - results_df['svm_cv_std'],
                 results_df['svm_cv_mean'] + results_df['svm_cv_std'], alpha=0.2)
plt.title("Model Performance vs. Max Features")
plt.xlabel("Max Features")
plt.ylabel("Cross-Validation Accuracy")
plt.legend()
plt.show()

# Evaluate Metrics on Best Configuration (max_features=5000)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(data['sentence']).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Naive Bayes
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)
nb_f1 = f1_score(y_test, nb_predictions, average='weighted')
print("Naive Bayes Performance:")
print("Accuracy:", accuracy_score(y_test, nb_predictions))
print("F1-Score:", nb_f1)

# SVM
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_f1 = f1_score(y_test, svm_predictions, average='weighted')
print("\nSVM Performance:")
print("Accuracy:", accuracy_score(y_test, svm_predictions))
print("F1-Score:", svm_f1)