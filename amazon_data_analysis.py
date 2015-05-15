### New file to munge / model
import pandas as pd
import matplotlib as plt
import pylab as pl
import numpy as np
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cross_validation import train_test_split
# %pylab inline



def parse(path):
    g = open(path, 'r')
    for l in g:
        yield eval(l)

def clean_columns():
    """ this drops columns that we mostly don't need.  It also converts the unixReviewTime to datetime"""

    df_meta.drop('categories', axis = 1, inplace = True)
    df_meta.drop('imUrl', axis = 1, inplace = True)
    df_meta.drop('related', axis = 1, inplace = True)
    df_meta.drop('salesRank', axis = 1, inplace = True)
    df_reviews.drop('reviewTime', axis = 1, inplace = True)
    df_reviews['unixReviewTime'] = pd.to_datetime(df_reviews['unixReviewTime'],unit='s')

def creating_basic_features():
    """This extracts information out of the tuple 'helpful' column so that we can start to create some other features"""
    df_reviews['helpful_votes'] = df_reviews.helpful.apply(lambda x: x[0])
    df_reviews['overall_votes'] = df_reviews.helpful.apply(lambda x: x[1])
    df_reviews['percent_helpful'] = df_reviews['helpful_votes'] / df_reviews['overall_votes']
    df_reviews['review_helpful'] = np.where((df_reviews.percent_helpful > .7) & (df_reviews.helpful_votes > 5), "Yes", "No")

def create_textblob_features():
    """uses the textblob module in order to extract features from the review text"""
    from textblob import TextBlob
    df_reviews['polarity'] = df_reviews['reviewText'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df_reviews['len_words'] = df_reviews['reviewText'].apply(lambda x: len(TextBlob(x).words))
    df_reviews['len_sentences'] = df_reviews['reviewText'].apply(lambda x: len(TextBlob(x).sentences))
    df_reviews['subjectivity'] = df_reviews['reviewText'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    df_reviews['words_per_sentence'] = df_reviews.len_words / df_reviews.len_sentences
    df_reviews['sentence_complexity'] = df_reviews.reviewText.apply(lambda x: float(len(set(TextBlob(x).words))) / float(len(TextBlob(x).words)))

# ok, let's read in the data
obj_reviews = parse('/Users/trevorsmith/Documents/metis/mcnulty_public/reviews_sports_outdoors_sample3.txt')
obj_meta = parse('/Users/trevorsmith/Documents/metis/mcnulty_public/meta_sports_outdoors_sample.txt')

# loop through every review and product and create list of dictionaries
reviews = []
for i in obj_reviews:
    reviews.append(i)

meta = []
for i in obj_meta:
    meta.append(i)

df_meta = pd.DataFrame(meta)
df_reviews = pd.DataFrame(reviews)

# now let's clean it and create basic features
clean_columns()
creating_basic_features()
# create_textblob_features()

# only taking reviews with at least 5 votes
df_updated = df_reviews[(df_reviews.overall_votes > 5)]


def model_accuracy(classifiers, steps):
    """This function takes a list of classifiers and outputs their accuracy, precision, and recall for every training split you specify.  The goal is to help the user find the optimal level of training / test data split for each model.  Be warned, this can take a while on large data sets

    This function takes two arguments:
    1) the classifiers you wish to use
        - example: ['MultinomialNB()', 'RandomForestClassifier()']
    2) the step incrementality for your training:
        - example: '.05'  This would split your training and test set starting at .05 and incrementing by .05 every time.  So the first pass would be testing the models using only 5% of the data as training.  The second pass would be using 10% of the data as training, etc."""


for x in np.arange(0.05,1, steps):
    classifiers = [MultinomialNB(), RandomForestClassifier(), LogisticRegression()]
    df_train, df_test = train_test_split(df_updated, test_size=x, random_state=0)
    vectorizer = CountVectorizer(min_df=1, stop_words='english',ngram_range=(0,6))
    X_train = vectorizer.fit_transform(df_train.reviewText)
    y_train = df_train.review_helpful
    X_test = vectorizer.transform(df_test.reviewText)
    y_test = df_test.review_helpful
    for i in classifiers:
        model = i
        model.fit(X_train, y_train)
        pred_values = model.predict(X_test)
        true_values = y_test
        ts = x
        recall = str(precision_recall_fscore_support(true_values == 'Yes', pred_values == 'Yes', average='binary')[1])
        precision = str(precision_recall_fscore_support(true_values == 'Yes', pred_values == 'Yes', average='binary')[0])
        accuracy = str(model.score(X_test, y_test))
        print str(model) + "training size: " + str(1-x)
        print "recall: " + str(recall) + "precision: " + str(precision) + "accuracy: " + str(accuracy)
