import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# Load the cleaned data
file_path = 'emotionalFiles/cleaned_sentiment_data.csv'
data = pd.read_csv(file_path)

# Add a sentence length column
data['sentence_length'] = data['sentence'].apply(len)

# Add a word count column
data['word_count'] = data['sentence'].apply(lambda x: len(x.split()))

# 1. Emotion distribution
emotion_counts = data['emotion'].value_counts()
print("Emotion Distribution:")
print(emotion_counts)

plt.figure(figsize=(8, 5))
sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette="viridis")
plt.title("Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.savefig('visualizations/emotion_distribution.png')
plt.show()  # Display the plot
plt.close()

# 2. Sentence length statistics
mean_length = data['sentence_length'].mean()
median_length = data['sentence_length'].median()
std_length = data['sentence_length'].std()
print("\nSentence Length Statistics:")
print(f"Mean: {mean_length}")
print(f"Median: {median_length}")
print(f"Standard Deviation: {std_length}")

# Violin and Stripplot of sentence lengths by emotion
plt.figure(figsize=(10, 6))
sns.violinplot(x='emotion', y='sentence_length', data=data, palette="Set3", inner=None)
sns.stripplot(x='emotion', y='sentence_length', data=data, color="black", alpha=0.5, jitter=True)
plt.title("Sentence Length Distribution by Emotion")
plt.savefig('visualizations/sentence_length_by_emotion_violin.png')
plt.show()  # Display the plot
plt.close()

# Word Count Distribution by Emotion
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='word_count', hue='emotion', kde=True, palette="coolwarm", bins=30)
plt.title("Word Count Distribution by Emotion")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.savefig('visualizations/word_count_distribution.png')
plt.show()  # Display the plot
plt.close()

# Enhanced Heatmap for Correlation
plt.figure(figsize=(8, 6))
correlation = data[['sentence_length', 'word_count']].corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", cbar=True)
plt.title("Feature Correlation Heatmap")
plt.savefig('visualizations/feature_correlation_heatmap.png')
plt.show()  # Display the plot
plt.close()

# Scatterplot of Sentence Length vs. Word Count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='word_count', y='sentence_length', hue='emotion', palette="Dark2", data=data)
plt.title("Sentence Length vs. Word Count")
plt.xlabel("Word Count")
plt.ylabel("Sentence Length")
plt.savefig('visualizations/sentence_length_vs_word_count.png')
plt.show()  # Display the plot
plt.close()

# 3. Word Clouds for Positive and Negative Emotions
positive_words = ' '.join(data[data['emotion'] == 'positive']['sentence'])
negative_words = ' '.join(data[data['emotion'] == 'negative']['sentence'])

positive_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(positive_words)
negative_wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_words)

# Positive Emotion Word Cloud
plt.figure(figsize=(10, 6))
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Positive Emotions")
plt.savefig('visualizations/positive_emotion_wordcloud.png')
plt.show()  # Display the plot
plt.close()

# Negative Emotion Word Cloud
plt.figure(figsize=(10, 6))
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Negative Emotions")
plt.savefig('visualizations/negative_emotion_wordcloud.png')
plt.show()  # Display the plot
plt.close()

# Top Words Visualization for Positive and Negative Emotions
positive_word_freq = Counter(positive_words.split()).most_common(20)
negative_word_freq = Counter(negative_words.split()).most_common(20)

positive_df = pd.DataFrame(positive_word_freq, columns=['word', 'count'])
negative_df = pd.DataFrame(negative_word_freq, columns=['word', 'count'])

# Positive Words Bar Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='count', y='word', data=positive_df, palette='viridis')
plt.title("Top Words in Positive Sentences")
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.savefig('visualizations/top_positive_words.png')
plt.show()  # Display the plot
plt.close()

# Negative Words Bar Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='count', y='word', data=negative_df, palette='Reds')
plt.title("Top Words in Negative Sentences")
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.savefig('visualizations/top_negative_words.png')
plt.show()  # Display the plot
plt.close()

print("Visualizations generated and saved to the 'visualizations' directory.")
