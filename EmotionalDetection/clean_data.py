import pandas as pd

# Load the dataset
file_path = 'emotionalFiles/combined_sentiment_data.csv'
data = pd.read_csv(file_path)

# Display the first few rows
print("Initial Dataset Preview:")
print(data.head())

# step 0
print("Dataset Columns:")
print(data.columns)


# Step 1: Remove rows with missing values
data.dropna(inplace=True)


# Step 2: Validate and clean the 'emotion' column
valid_emotions = ['positive', 'negative']
data = data[data['emotion'].isin(valid_emotions)]

# Step 3.1: Remove duplicates if any
data.drop_duplicates(inplace=True)
# Step 3.2: Remove duplicates based only on the 'sentence' column
data.drop_duplicates(subset='sentence', inplace=True)

# Display the cleaned dataset information
print("\nCleaned Dataset Information:")
print(data.info())

# Display basic statistics
print("\nBasic Statistics:")
print(data.describe())

# Save the cleaned data to a new CSV file
cleaned_file_path = 'emotionalFiles/cleaned_sentiment_data.csv'
data.to_csv(cleaned_file_path, index=False)

print(f"\nCleaned data saved to: {cleaned_file_path}")
