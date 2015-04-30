### Error Challenges
## Challenge 1
# read in data
import pandas as pd
import numpy as np
from patsy import dmatrices
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pylab as pl
from sklearn.learning_curve import learning_curve
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


headers_all = ['party', 'handicapped_infants',
'water_project_cost_sharing','adoption_of_the_budget_resolution','physician_fee_freeze','el_salvador_aid', 'religious_groups_in_schools','anti_satellite_test_ban','aid_to_nicraguan_contras', 'mx_missile', 'immigration','synfuels_corporation_cutback', 'education_spending', 'superfund_right_to_sue','crime', 'duty_free_exports', 'export_administration_act_south_africa']
df = pd.read_csv('data.csv', names = headers_all)

# converting the data
df = df.replace('y', 1).replace('n', 0).replace('?', np.nan)

# there are a lot of '?' in the data
# goal is to replace them with column mean
df = df.fillna(df.mean())

# manually add the intercept
df['intercept'] = 1.0
train_cols = df.columns[1:]

features = ['handicapped_infants',
'water_project_cost_sharing','adoption_of_the_budget_resolution','physician_fee_freeze','el_salvador_aid', 'religious_groups_in_schools','anti_satellite_test_ban','aid_to_nicraguan_contras', 'mx_missile', 'immigration','synfuels_corporation_cutback', 'education_spending', 'superfund_right_to_sue','crime', 'duty_free_exports', 'export_administration_act_south_africa']

# split data
df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)

# bayes model
bayes = GaussianNB()
bayes.fit(df_train[features], df_train['party'])
preds = bayes.predict(df_test[features])
accuracy = np.where(preds==df_test['party'], 1, 0).sum() / float(len(df_test))
print "Bayes: Accuracy: %3f" % (accuracy)

print precision_recall_fscore_support('democrat'==true_values, 'democrat'==pred_values, average='binary')

## Challenge 2
