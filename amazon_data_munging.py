import pandas as pd
import matplotlib as plt
import pylab as pl
%matplotlib inline
%pylab inline

def parse(path):
  g = open(path, 'r')
  for l in g:
    yield eval(l)

obj_reviews = parse('/Users/trevorsmith/Documents/metis/mcnulty_public/reviews_sports_outdoors_sample.txt')
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

# dropping columns we don't need
df_meta.drop('categories', axis = 1, inplace = True)
df_meta.drop('imUrl', axis = 1, inplace = True)
df_meta.drop('related', axis = 1, inplace = True)
df_meta.drop('salesRank', axis = 1, inplace = True)
df_reviews.drop('reviewTime')

df_reviews['unixReviewTime'] = pd.to_datetime(df_reviews['unixReviewTime'],unit='s')

# checking the distribution of review scores
df_reviews['overall'].hist(bins = 5)

# gotta format the helpful column
df_reviews['helpful_votes'] = df_reviews.helpful.apply(lambda x: x[0])
df_reviews['overall_votes'] = df_reviews.helpful.apply(lambda x: x[1])

# looking at most unhelpful!
df_reviews['unhelpfulness'] = df_reviews.overall_votes - df_reviews.helpful_votes
x = df_reviews.sort("unhelpfulness", ascending=False).head()

# let's create another column just for fun :P
df_reviews['percent_helpful'] = df_reviews['helpful_votes'] / df_reviews['overall_votes']

# plotting distribution of percent helpful
df_reviews['percent_helpful'].hist(bins = 9).show()

# create column to say if review was helpful or not
# set threshold at 50%
df_reviews['review_helpful'] = np.where(df_reviews['percent_helpful'] >.5, 'Yes', 'No')

# unhelpfulness LOL
print x.reviewText.iloc[0]

# sentiment analysis stuff
from textblob import TextBlob

# need to analyze missing values more
# dropping na's cut data set size in half!
# df_reviews = df_reviews.dropna()


# some feature engineering for things I think will predict
df_reviews['polarity'] = df_reviews['reviewText'].apply(lambda x: TextBlob(x).sentiment.polarity)
df_reviews['len_words'] = df_reviews['reviewText'].apply(lambda x: len(TextBlob(x).words))
df_reviews['len_sentences'] = df_reviews['reviewText'].apply(lambda x: len(TextBlob(x).sentences))
df_reviews['subjectivity'] = df_reviews['reviewText'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# ok let's plot their distributions
df_reviews.polarity.hist(bins = 20)
# normal
df_reviews.subjectivity.hist(bins = 20)
# normal
df_reviews.len_words.hist(bins = 20)
# left skew
df_reviews.len_sentences.hist(bins = 20)
# left skew
# log transforms may be needed for the last two variables
df_reviews.len_words = df_reviews.len_words.apply(np.log)
df_reviews.len_sentences = df_reviews.len_sentences.apply(np.log)

# what does the percent helpful variable look like?
df_reviews.percent_helpful.hist(bins = 20)
# a little over 5k reviews not helpful at all
# almost 25k reviews 100% helpful...interesting patttern

# ok now to test some MODELS
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot

features = ['overall', 'polarity', 'subjectivity', 'len_sentences', 'len_words']

# let's drop reviews with no text
import numexpr
df_reviews = df_reviews.query('len_words > 0')

# create matrices to analyze
X = df_reviews[features]
y = df_reviews['review_helpful']

# decision tree
dt_clf = DecisionTreeClassifier()
scores = cross_validation.cross_val_score(dt_clf, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# random forest
rf_clf = RandomForestClassifier()
scores = cross_validation.cross_val_score(rf_clf, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# support vector machine
svm_clf = SVC()
scores = cross_validation.cross_val_score(svm_clf, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# naive bayes
bayes_clf = GaussianNB()
scores = cross_validation.cross_val_score(bayes_clf, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

df_reviews['words_per_sentence'] = df_reviews.len_words / df_reviews.len_sentences
features2 = ['overall', 'polarity', 'subjectivity', 'len_sentences', 'len_words', 'words_per_sentence']

X2 = df_reviews[features2]
rf_clf = RandomForestClassifier()
scores = cross_validation.cross_val_score(rf_clf, X2, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# plots for dayz
df_reviews.plot(kind='scatter', x='percent_helpful', y='helpful_votes')
df_reviews.plot(kind='scatter', x='percent_helpful', y='len_words')
df_reviews.plot(kind='scatter', x='percent_helpful', y='len_sentences')
df_reviews.plot(kind='scatter', x='percent_helpful', y='polarity')
df_reviews.plot(kind='scatter', x='percent_helpful', y='subjectivity')
df_reviews.plot(kind='scatter', x='percent_helpful', y='words_per_sentence')
df_reviews.len_words.hist(bins=100)
df_reviews.plot(kind='scatter', x='len_words', y='words_per_sentence')
df_reviews.plot(kind='scatter', x='len_words', y='len_sentences')
df_reviews.plot(kind='scatter', x='len_sentences', y='words_per_sentence')
df_reviews.plot(kind='scatter', x='percent_helpful', y='words_per_sentence')

# ok now let's groupby
df_temp = df_reviews[['asin','unixReviewTime']]
df_temp2 = df_temp.groupby('asin').min()

df_temp2.reset_index(inplace=True)
df_temp.rename(columns={'unixReviewTime':'firstreviewdate'}, inplace=True)

# merge back data sets
df_new = pd.merge(df_reviews, df_temp2, how='left', on='asin' )

# let's add in time element
df_new['time_from_first_review'] = df_new.unixReviewTime_x - df_new.unixReviewTime_y
df_new.time_from_first_review = df_new.time_from_first_review.astype(int)
df_new.plot(kind='scatter', x='time_from_first_review', y='helpful_votes')
df_new.plot(kind='scatter', x='time_from_first_review', y='overall_votes')
features3 = ['overall', 'polarity', 'subjectivity', 'len_sentences', 'len_words', 'words_per_sentence', 'time_from_first_review']
X3 = df_new[features3]
y3 = df_new.review_helpful

rf_clf = RandomForestClassifier()
scores = cross_validation.cross_val_score(rf_clf, X3, y3, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

bayes_clf = GaussianNB()
scores = cross_validation.cross_val_score(bayes_clf, X3, y3, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# setting different thresholds for a 'helpful' review
df_new.review_helpful = np.where((df_new.percent_helpful > .7) & (df_new.helpful_votes > 5), "Yes", "No")

