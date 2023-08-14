# KEYWORD-MATCHING APPROACH WITH ALL DATASETS
# in this code the size of n-grams and list is suppossed to be manually changed depending on the desired experiment
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import nltk
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import random
from nltk.stem import PorterStemmer
import seaborn as sns
import matplotlib.pyplot as plt
ps = PorterStemmer()
from sklearn.metrics import classification_report

# Calculate precision, recall, and F1-score using classification_report

# download stopwords and wordnet
nltk.download('stopwords')
nltk.download('wordnet')
# read in the dataset
data = pd.read_csv('wdutch_train.txt')
test_data = pd.read_csv('wdutch_test.txt')
# preprocess text data
stop_words = set(stopwords.words('english'))
# Update the stop_words set with the words to be added
# stop_words.update(['competence available','competence information', 'information available', 'availableknowledge information',
#                    'information availableknowledge', 'availableskills information', 'information availableskills'])
lemmatizer = WordNetLemmatizer()
# Modify the preprocess_text function to exclude the added words
def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Remove numbers
    tokens = [token for token in tokens if not re.match(r'\d+', token)]
    # Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    # Remove additional words
    additional_words = [':', ':', '-', 'knowledge', 'available', 'information'
     'competence',' competence', 'availablecompetences', 'availableknowledge', 'availableskills']

    # consider this again after changing the data-stoprwords list

    tokens = [t for t in tokens if t.strip() not in additional_words]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    # Re-join tokens into a string
    return ' '.join(tokens)


data['description'] = data['description'].apply(preprocess_text)

# select the column containing the preprocessed text data
text_data = data['description']
# print(text_data)
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# create tf-idf vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2))

# fit and transform the text data to a tf-idf matrix
tfidf_matrix = vectorizer.fit_transform(text_data)

# get the feature names (words and bigrams)
feature_names = vectorizer.get_feature_names_out()
# print(feature_names)
# create a dictionary to store the top words for each document class
top_words_by_class = {}

# iterate over each unique document class (eqf_level_id)
for eqf_level in data['eqf_level_id'].unique():
    # get the indices of the documents that belong to this class
    class_docs = data[data['eqf_level_id'] == eqf_level].index
    # create a list to store the tf-idf scores for each word
    word_scores = []
    # iterate over each document in this class
    # https://stackoverflow.com/questions/34449127/sklearn-tfidf-transformer-how-to-get-tf-idf-values-of-given-words-in-documen
    for doc_idx in class_docs:
        # print(doc_idx)
        doc = tfidf_matrix[doc_idx]
        feature_index = doc.indices
        tfidf_scores = doc.data
        # append the tf-idf scores for this document to the list
        word_scores.append(dict(zip([feature_names[i] for i in feature_index], tfidf_scores)))
    # combine the tf-idf scores for all documents in this class
    combined_scores = {}
    for doc_scores in word_scores:
        for word, score in doc_scores.items():
            # print(word)
            if word in combined_scores:
                combined_scores[word] += score
            else:
                combined_scores[word] = score
    # 81-87  calculating the classes which has more words and this explaings class 0s


    # sort the words by their combined tf-idf scores
    #By sorting the words based on their tf-idf scores and selecting the top words, we are prioritizing
    # the words that have the highest scores and, therefore,
    # the greatest potential to contribute to the classification of a document into a specific class.
    sorted_words = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    # store the top 50/100 etc. words for this class in the dictionary
    top_words_by_class[eqf_level] = [w[0] for w in sorted_words[:1000]]

# Preprocess the text data in the test dataset
test_data['description'] = test_data['description'].apply(preprocess_text)

# Select the column containing the preprocessed text data from the test dataset
test_text_data = test_data['description']

# Transform the preprocessed text data of the test dataset into TF-IDF vectors
test_tfidf_matrix = vectorizer.transform(test_text_data)

# Store the actual classes from the test dataset
actual_classes = test_data['eqf_level_id']
actual_classes=actual_classes.to_list()
# Initialize variables for accuracy calculation
correct_predictions = 0
total_predictions = 0
predicted_classes = []

# Classify the sentences from the test dataset and check accuracy fixted by using ChatGpt for the classification by list part
for i, sentence in enumerate(test_text_data):
    sentence_tfidf = test_tfidf_matrix[i]
    predicted_class = None
    max_score = 0
    matching_words_by_class = {}

    for eqf_level in data['eqf_level_id'].unique():
        class_docs = data[data['eqf_level_id'] == eqf_level].index
        class_tfidf = tfidf_matrix[class_docs].mean(axis=0)
        score = (sentence_tfidf @ class_tfidf.T)[0, 0]

        # Get the top words for the current class
        top_words = top_words_by_class[eqf_level]
        # Calculate the number of matching words between the sentence and the top words of the class
        matching_words = set(preprocess_text(sentence).split()) & set(top_words)
        # Print the matching words for the current class
        # if matching_words:
        #     print(f"Sentence: {sentence}")
        #     print(f"Matching words for class {eqf_level}: {', '.join(matching_words)}")
        #     print("----------")

        # Calculate the weighted score by multiplying the original score with the ratio of matching words to the total top words
        weighted_score = score * (len(matching_words) / len(top_words))

        matching_words_by_class[eqf_level] = weighted_score
        if weighted_score > max_score:
            max_score = weighted_score
            predicted_class = eqf_level

    if predicted_class is None:
        predicted_class = random.randint(1, 8)

    if predicted_class == actual_classes[i]:
        correct_predictions += 1

    total_predictions += 1
    predicted_classes.append(predicted_class)

    # if predicted_class==2:
        # print('x')
# print(len(predicted_classes))
    # print(f"Sentence: {sentence}")
    # print(f"Predicted class: {predicted_class}")
    # print(f"Actual class: {actual_classes[i]}")
    # print("----------")
accuracy = correct_predictions / total_predictions

# Calculate precision, recall, and F1-score
precision = precision_score(actual_classes, predicted_classes, average='weighted', zero_division=0)
recall = recall_score(actual_classes, predicted_classes, average='weighted', zero_division=0)
f1_score = 2 * (precision * recall) / (precision + recall)

# Generate the classification report
class_report = classification_report(actual_classes, predicted_classes, zero_division=0)

# Print accuracy, precision, recall, and F1-score
print('actual_classes', '\n', actual_classes, 'predicted_classes', type(predicted_classes))
print('PRE', predicted_classes, 'ACT', actual_classes)
print('list size 55 - Keyword-matching approach with unigrams all data')
print(f"Accuracy: {accuracy}")
print(f"P: {precision}")
print(f"R: {recall}")
print(f"F1: {f1_score}")

# Initialize the confusion matrix
conf_matrix = confusion_matrix(actual_classes, predicted_classes)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)