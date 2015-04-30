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

## Challenge 6
# code above already does this

## Challenge 7
from sklearn.learning_curve import learning_curve
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

model = LogisticRegression()
X = df[features]
y = df['party']

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

plot_learning_curve(model, 'Logistic Regression', X, y, ylim=None, cv=5,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5))

## Challenge 8
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Bayes
bayes = GaussianNB()
bayes.fit(df_train[features], df_train['party'])
preds = bayes.predict(df_test[features])
accuracy = np.where(preds==df_test['party'], 1, 0).sum() / float(len(df_test))
print "Bayes: Accuracy: %3f" % (accuracy)

# Support Vector Machine
svm = SVC()
svm.fit(df_train[features], df_train['party'])
preds = svm.predict(df_test[features])
accuracy = np.where(preds==df_test['party'], 1, 0).sum() / float(len(df_test))
print "SVM: Accuracy: %3f" % (accuracy)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(df_train[features], df_train['party'])
preds = dt.predict(df_test[features])
accuracy = np.where(preds==df_test['party'], 1, 0).sum() / float(len(df_test))
print "Decision Tree: Accuracy: %3f" % (accuracy)

# Random Forest
rf = RandomForestClassifier()
rf.fit(df_train[features], df_train['party'])
preds = rf.predict(df_test[features])
accuracy = np.where(preds==df_test['party'], 1, 0).sum() / float(len(df_test))
print "Random Forest: Accuracy: %3f" % (accuracy)

## Challenge 9
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation

# we will be performing 5 fold cross validation

# logistic regression
logit_clf = LogisticRegression()
scores = cross_validation.cross_val_score(logit_clf, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# k nearest neighbours
knn_clf = KNeighborsClassifier(n_neighbors=21)
scores = cross_validation.cross_val_score(knn_clf, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# naive bayes
bayes_clf = GaussianNB()
scores = cross_validation.cross_val_score(bayes_clf, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# support vector machines
svm_clf = SVC()
scores = cross_validation.cross_val_score(svm_clf, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# decision trees
dt_clf = DecisionTreeClassifier()
scores = cross_validation.cross_val_score(dt_clf, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# random forest
rf_clf = RandomForestClassifier()
scores = cross_validation.cross_val_score(rf_clf, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))




