Sentiment Analysis with Naive Bayes and Support Vector Machine

Overview

This project performs sentiment analysis on a dataset of labeled sentences. Using text vectorization with TF-IDF, two machine learning models are trained and evaluated: Naive Bayes and Support Vector Machine (SVM). The analysis includes performance metrics, confusion matrices, and a comparison of the models' accuracy. Additionally, it offers insights into incorrect predictions for further improvement.

Project Structure

The project contains the following main components:

cleaned_sentiment_data.csv: The dataset used for training and evaluation. It contains labeled sentences with their associated emotions.

Python Script: The script implements the pipeline from data preparation to model evaluation.

Output Files:

predictions/Naive_Bayes_correct_predictions.csv: Correct predictions made by the Naive Bayes model.

predictions/Naive_Bayes_incorrect_predictions.csv: Incorrect predictions made by the Naive Bayes model.

predictions/SVM_correct_predictions.csv: Correct predictions made by the SVM model.

predictions/SVM_incorrect_predictions.csv: Incorrect predictions made by the SVM model.

Visualizations:

Confusion matrices for both Naive Bayes and SVM models.


Setup Instructions

Install Required Libraries:
Make sure the following Python libraries are installed:

pip install pandas numpy scikit-learn seaborn matplotlib

Directory Structure:
Ensure the following directory structure is in place:

project-directory/
|-- emotionalFiles/
|   |-- cleaned_sentiment_data.csv
|-- predictions/
|   |-- (Generated prediction files will be saved here)
|-- script.py (Main Python script)

Run the Script:
Execute the script in your Python environment:

python predictor.py

Features

1. Data Preprocessing

Text data is vectorized using the TF-IDF Vectorizer with a maximum of 5000 features.

Data is split into training and testing sets with a stratified split to balance emotions.

2. Model Implementation

Naive Bayes:

Simple and effective for text classification.

Fast to train and test.

Support Vector Machine (SVM):

Linear kernel used for robust classification.

Handles high-dimensional data well.

3. Performance Evaluation

Confusion Matrices for each model.

Classification Report showing precision, recall, and F1-score.

Accuracy Comparison between models.

4. Insights into Predictions

Separate CSV files are generated for correct and incorrect predictions for each model, enabling error analysis.

5. Visualizations

Heatmaps of confusion matrices for both models.

A bar chart comparing the accuracy of Naive Bayes and SVM.

Improvements and Suggestions

Additional Features:

Incorporate bigrams or trigrams into the TF-IDF vectorizer.

Experiment with other models like Random Forest or Gradient Boosting.

Hyperparameter Tuning:

Optimize parameters for both Naive Bayes and SVM for better performance.

Error Analysis:

Perform in-depth analysis of incorrect predictions to identify patterns.

Data Augmentation:

Expand the dataset with synonyms or paraphrased sentences for better generalization.

Visualization:

Add sentiment distributions or word clouds to explore frequent terms for each emotion.

Results

Naive Bayes achieved an accuracy of X%.

SVM achieved an accuracy of Y%.

Based on the results, [model choice] performed better for this dataset.
