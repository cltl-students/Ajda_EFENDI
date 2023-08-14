from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.stem import PorterStemmer
ps = PorterStemmer()

# download stopwords and wordnet
nltk.download('stopwords')
nltk.download('wordnet')

# read in the dataset
data = pd.read_csv('merged_training.txt')

# preprocess text data
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

# Load the training and test datasets- depending on the system the training and testing datasets should be uncommented
train_data = pd.read_csv('merged_training.txt')
# train_data=pd.read_csv('sec_merged_training.txt')
# train_data=pd.read_csv('wdutch_train.txt')
test_data = pd.read_csv('NEW_DUTCH')
# test_data=pd.read_csv('wdutch_test.txt')

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

# Create TF-IDF vectorizer with unigram and bigram features
vectorizer = TfidfVectorizer(ngram_range=(3, 3))
X_train = vectorizer.fit_transform(train_data['description'])
X_test = vectorizer.transform(test_data['description'])

# Define the target variable
y_train = train_data['eqf_level_id']
y_test = test_data['eqf_level_id']

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate overall performance metrics
precision_weighted = precision_score(y_test, y_pred, average='weighted')
recall_weighted = recall_score(y_test, y_pred, average='weighted')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

# Print overall performance metrics
print(f"Precision (Weighted Avg): {precision_weighted:.4f}")
print(f"Recall (Weighted Avg): {recall_weighted:.4f}")
print(f"F1-Score (Weighted Avg): {f1_weighted:.4f}")

# Calculate confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_mat)
