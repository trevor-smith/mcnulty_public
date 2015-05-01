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
df_reviews['percent helpful'] = df_reviews['helpful'] / df_reviews['overall']
df_reviews['percent helpful'] = df_reviews['percent helpful'].clip_upper(1)

# plotting distribution of percent helpful
df_reviews['percent helpful'].hist(bins = 9)

# create column to say if review was helpful or not
# set threshold at 40%
df_reviews['review helpful?'] = np.where(df_reviews['percent helpful'] >.4, 'Yes', 'No')


