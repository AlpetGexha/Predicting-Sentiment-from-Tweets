# %% md
# # Predicting Sentiment from Tweets
# 
# ### Project Overview:
# Sentiment analysis is a powerful tool used to understand public sentiment and opinion from social media content, particularly on platforms like Twitter. This project aims to apply machine learning techniques to classify tweets as either positive or negative. By doing so, businesses, researchers, and individuals can better understand public sentiment, helping inform decisions and strategies.
# 
# The project utilizes the Twitter dataset provided by the NLTK library, which contains 10,000 labeled tweets. These tweets are evenly divided between positive and negative sentiments, providing a balanced dataset for training and evaluation.
# %%
from nltk.corpus import twitter_samples  # sample Twitter dataset from NLTK

# nltk.download('twitter_samples')
# select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

print('Number of positive tweets: ', len(all_positive_tweets))
print('Number of negative tweets: ', len(all_negative_tweets))
# %% md
# # Clean and Preprocess Tweets
# %%
import re


def clean_tweet(tweet):
    # Remove punctuation and make lowercase
    tweet = re.sub(r'[^\w\s]', '', tweet.lower())

    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet)

    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'@\w+|#', '', tweet)

    return tweet


# %%
from nltk.corpus import stopwords

# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def remove_stopwords(tweet):
    return ' '.join([word for word in tweet.split() if word not in stop_words])


# %%
from nltk.stem import PorterStemmer


def stem_tweet(tweet):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in tweet.split()])


# %%
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def lemmatize_tweet(tweet):
    return lemmatizer.lemmatize(tweet, 'v')


# %%
def preprocess_tweet(tweet):
    tweet = clean_tweet(tweet)
    tweet = remove_stopwords(tweet)
    # tweet = stem_tweet(tweet)
    tweet = lemmatize_tweet(tweet)

    return tweet


# %%
print(all_positive_tweets[6])
preprocess_tweet(all_positive_tweets[6])
# %%
positive_tweets = [preprocess_tweet(tweet) for tweet in all_positive_tweets]
negative_tweets = [preprocess_tweet(tweet) for tweet in all_negative_tweets]
# %% md
# # Exploratory Data Analysis
# %%
import matplotlib.pyplot as plt

# Calculate tweet lengths for both classes
pos_lengths = [len(tweet.split()) for tweet in all_positive_tweets]
neg_lengths = [len(tweet.split()) for tweet in all_negative_tweets]

# Plotting
plt.figure(figsize=(12, 6))
plt.hist(pos_lengths, bins=30, alpha=0.7, label='Positive', color='blue')
plt.hist(neg_lengths, bins=30, alpha=0.7, label='Negative', color='red')
plt.title('Tweet Length Distribution by Sentiment')
plt.xlabel('Tweet Length (Number of Words)')
plt.ylabel('Frequency')
plt.legend()
plt.show()
# %%
from wordcloud import WordCloud

# Positive Word Cloud
pos_text = ' '.join(positive_tweets)
pos_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(pos_text)
plt.figure(figsize=(10, 5))
plt.imshow(pos_wordcloud, interpolation='bilinear')
plt.title('Positive Tweet Word Cloud')
plt.axis('off')
plt.show()

# Negative Word Cloud
neg_text = ' '.join(negative_tweets)
neg_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(neg_text)
plt.figure(figsize=(10, 5))
plt.imshow(neg_wordcloud, interpolation='bilinear')
plt.title('Negative Tweet Word Cloud')
plt.axis('off')
plt.show()
# %%

# count all the word on positive and negative tweets
from collections import Counter

# Count all the words in positive tweets
pos_words = ' '.join(positive_tweets)
pos_words = pos_words.split()
pos_word_count = Counter(pos_words)

# Count all the words in negative tweets
neg_words = ' '.join(negative_tweets)
neg_words = neg_words.split()
neg_word_count = Counter(neg_words)

print('Unique Positive words: ', len(pos_word_count))
print('Unique Negative words: ', len(neg_word_count))

print("Most common positive words: ", pos_word_count.most_common(5))
print("Most common negative words: ", neg_word_count.most_common(5))
# %% md
# # Train and Test Data Split
# %%
from sklearn.model_selection import train_test_split

# Combine tweets and labels
all_tweets = positive_tweets + negative_tweets
labels = [1] * len(positive_tweets) + [0] * len(negative_tweets)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_tweets, labels, test_size=0.2, random_state=42)
# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def vectorizer(vectorizer_name):
    if vectorizer_name == 'Tfidf':
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    elif vectorizer_name == 'Count':
        vectorizer = CountVectorizer()
    else:
        raise ValueError("Invalid vectorizer name")

    return vectorizer


# %%

# vectorizer = TfidfVectorizer(
#     max_features=5000,
#     # ngram_range=(1,2)
# )
vectorizer = vectorizer('Count')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(X_test_tfidf)
# %% md
# # Model Training and Evaluation
# %%
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# Configurable model choices
def train_model(model_name='LogisticRegression', X_train=X_train_tfidf, y_train=y_train):
    model_dict = {'MultinomialNB': MultinomialNB(), 'BernoulliNB': BernoulliNB(),
        'LogisticRegression': LogisticRegression(max_iter=200), 'SVM': SVC(kernel='linear', probability=True),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)}

    if model_name not in model_dict:
        raise ValueError("Invalid model name")

    # Train the selected model
    model = model_dict[model_name]
    model.fit(X_train, y_train)

    return model


# %%
from sklearn.metrics import accuracy_score, classification_report

# Initialize and train the model
model = train_model('SVM')
# %%
# Predict on the test set
y_pred_test = model.predict(X_test_tfidf)
y_pred_train = model.predict(X_train_tfidf)
# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_train, y_pred_train)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.show()
# %%
accuracy = accuracy_score(y_train, y_pred_train)
print(f"Train Data Accuracy: {accuracy * 100:.2f}%")

test_data_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test Data Accuracy: {test_data_accuracy * 100:.2f}%")

# Classification report
print("Classification Report:")
print(classification_report(y_train, y_pred_train))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_train, y_pred_train))
