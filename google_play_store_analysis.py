# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the datasets
data = pd.read_csv("apps.csv")
reviews_data = pd.read_csv("user_reviews.csv")

# Check the first few rows of the app data
print("App Data:")
print(data.head())

# Check the first few rows of the reviews data
print("\nReviews Data:")
print(reviews_data.head())

# Step 1: Data Cleaning (For both datasets)
# Cleaning the app data
data = data.dropna()  # Remove rows with missing values
data['Price'] = data['Price'].replace('[\$,]', '', regex=True).astype(float)
data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')

# Cleaning the reviews data
reviews_data = reviews_data.dropna()  # Remove rows with missing reviews
reviews_data['Translated_Review'] = reviews_data['Translated_Review'].astype(str)


# Merge the two datasets on the 'App' column (assuming they share this column)
merged_data = pd.merge(data, reviews_data, on='App')

# Step 2: Category Exploration
category_counts = data['Category'].value_counts()

# Visualize category distribution
plt.figure(figsize=(10, 6))
sns.countplot(data['Category'], order=category_counts.index)
plt.title("Distribution of Apps by Category")
plt.xticks(rotation=90)
plt.show()

# Step 3: Metrics Analysis (for app ratings and price trends)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Rating'], y=data['Price'])
plt.title('Rating vs Price of Apps')
plt.xlabel('Rating')
plt.ylabel('Price')
plt.show()

# Step 4: Sentiment Analysis on Reviews
# Use the VADER Sentiment Analyzer on reviews
analyzer = SentimentIntensityAnalyzer()

# Example sentiment analysis on a few reviews
sample_reviews = reviews_data['Translated_Review'].head(10)
for review in sample_reviews:
    sentiment_score = analyzer.polarity_scores(str(review))
    print(f"Review: {review}\nSentiment: {sentiment_score}\n")

# Step 5: Interactive Visualization (Optional)
import plotly.express as px

fig = px.bar(category_counts, x=category_counts.index, y=category_counts.values, title="App Distribution by Category")
fig.show()
