## Challenge 1
# read in data
import pandas as pd
import numpy as np
from patsy import dmatrices
import matplotlib.pyplot as plt
import statsmodels.api as sm

headers_all = ['party', 'handicapped_infants',
'water_project_cost_sharing','adoption_of_the_budget_resolution','physician_fee_freeze','el_salvador_aid', 'religious_groups_in_schools','anti_satellite_test_ban','aid_to_nicraguan_contras', 'mx_missile', 'immigration','synfuels_corporation_cutback', 'education_spending', 'superfund_right_to_sue','crime', 'duty_free_exports', 'export_administration_act_south_africa']
df = pd.read_csv('data.csv', names = headers_all)

# converting the data
df = df.replace('y', 1).replace('n', 0).replace('?', np.nan)

## Challenge 2
# there are a lot of '?' in the data
# goal is to replace them with column mean
df = df.fillna(df.mean())

# manually add the intercept
df['intercept'] = 1.0
train_cols = df.columns[1:]

# split data into training and test
from sklearn.cross_validation import train_test_split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)

## Challenge 3
# Using scikit.learn's KNN algorithm, train a model that predicts the party

from sklearn.neighbors import KNeighborsClassifier
import pylab as pl

features = ['handicapped_infants',
'water_project_cost_sharing','adoption_of_the_budget_resolution','physician_fee_freeze','el_salvador_aid', 'religious_groups_in_schools','anti_satellite_test_ban','aid_to_nicraguan_contras', 'mx_missile', 'immigration','synfuels_corporation_cutback', 'education_spending', 'superfund_right_to_sue','crime', 'duty_free_exports', 'export_administration_act_south_africa']

# this uses the training set to test accuracy
results = []
for n in range(1, 20, 1):
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(df_train[features], df_train['party'])
    preds = clf.predict(df_train[features])
    accuracy = np.where(preds==df_train['party'], 1, 0).sum() /float(len(df_train))
    print "Neighbors: %d, Accuracy: %3f" % (n, accuracy)

    results.append([n, accuracy])

results = pd.DataFrame(results, columns=["n", "accuracy"])

pl.plot(results.n, results.accuracy)
pl.title("Accuracy with Increasing K")
pl.show()

# this uses the test set to test accuracy
results = []
for n in range(1, 20, 1):
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(df_train[features], df_train['party'])
    preds = clf.predict(df_test[features])
    accuracy = np.where(preds==df_test['party'], 1, 0).sum() /float(len(df_test))
    print "Neighbors: %d, Accuracy: %3f" % (n, accuracy)

    results.append([n, accuracy])

results = pd.DataFrame(results, columns=["n", "accuracy"])

pl.plot(results.n, results.accuracy)
pl.title("Accuracy with Increasing K")
pl.show()

# test1 =  pd.crosstab(df['party'], df['handicapped_infants'], rownames=['party'])

## Challenge 4
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
train_model = model.fit(df_train[features], df_train['party'])
print train_model
print "the train accuracy is " + str(train_model.score(df_train[features], df_train['party']))
print "the test accuracy is " + str(train_model.score(df_test[features], df_test['party']))

## Challenge 5
republicans = float((df['party'] == 'republican').sum())
democrats = float((df['party'] == 'democrat').sum())
x = [1, 2]
y = [democrats, republicans]
labels = ['democrat', 'republican']

plt.bar(x, y, color="blue", align = 'center')
plt.xticks(x, labels)



