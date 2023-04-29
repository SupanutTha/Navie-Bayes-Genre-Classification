# Navie-Bayes-Genre-Classification

This code is used to train and evaluate a Naive Bayes classifier on a dataset of English song lyrics in order to predict the genre of a song.

## Dependencies
The following packages are required to run the code:

pandas
nltk
contractions
re
wordcloud
matplotlib
sklearn

## Dataset
I use english_cleaned_lyrics.csv from [hiteshyalamanchili/SongGenreClassification](https://github.com/hiteshyalamanchili/SongGenreClassification/tree/master/dataset)


## Preprocessing
The lyrics of each song in the dataset are preprocessed in order to remove stop words, contractions, and special characters. The resulting filtered lyrics are stored in a new column called "Filtered Lyrics".

## Training
The code trains a Naive Bayes classifier to predict the genre of a song based on its filtered lyrics. The training and testing data are split using the train_test_split function from the sklearn library.

## Evaluation
The code evaluates the accuracy of the trained classifier on the testing data using the accuracy_score function from the sklearn library. It also predicts the genre of each song in the testing set and prints the result in a pandas DataFrame.
