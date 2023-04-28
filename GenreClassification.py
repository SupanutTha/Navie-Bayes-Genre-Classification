import pandas as pd
import nltk
from nltk.corpus import stopwords
# from Orange.data.pandas_compat import table_from_frame,table_to_frame
from contractions import fix
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Download the stop words list from NLTK
nltk.download('stopwords')

# Load the data from a CSV file
df = pd.read_csv('english_cleaned_lyrics.csv')

# Define a function to count the number of words in a string, excluding stop words
stop_words = set(stopwords.words('english'))
def count_words_without_stopwords(s):
    words = s.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return len(filtered_words)

def remove_stop_words(s):
    # Replace contractions with their expanded forms
    s = fix(s)
    # Remove special characters
    s = re.sub(r"'s\b", "", s) # Remove 's form
    s = re.sub(r"/", "", s) # Remove forward slashes
    s = re.sub(r'"', "", s) # Remove double quotes
    # Split the lyrics into words
    words = s.split()
    # Filter out stop words
    filtered_words = [word for word in words if word.lower() not in stop_words]
    # Join the remaining words back into a string
    return " ".join(filtered_words)

# Apply the remove_stop_words function to the "Lyrics" column to create a new column called "Filtered Lyrics"
df["Filtered Lyrics"] = df["lyrics"].apply(remove_stop_words)

# Define the pipeline
pipeline = Pipeline([
    ("vectorizer", CountVectorizer()), # Convert lyrics to a bag-of-words representation
    ("classifier", MultinomialNB()) # Train a Naive Bayes classifier to predict the genre
])

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df["Filtered Lyrics"], df["genre"], test_size=0.2, random_state=42)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Evaluate the model on the testing data
from sklearn.metrics import accuracy_score
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict the genre for each song in the testing set
y_pred_test = pipeline.predict(X_test)

# Combine the predicted genre with the actual genre for each song in the testing set
result_df = pd.DataFrame({
    "Lyrics": X_test,
    "Actual Genre": y_test,
    "Predicted Genre": y_pred_test
})

# Print the result of the prediction for each song in the testing set
print(result_df)
