# imports
import pandas as pd
import numpy as np
from patsy import dmatrices
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot

# reading in data, transforms, etc
# this was already done for previous set of challenges
headers_all = ['party', 'handicapped_infants',
'water_project_cost_sharing','adoption_of_the_budget_resolution','physician_fee_freeze','el_salvador_aid', 'religious_groups_in_schools','anti_satellite_test_ban','aid_to_nicraguan_contras', 'mx_missile', 'immigration','synfuels_corporation_cutback', 'education_spending', 'superfund_right_to_sue','crime', 'duty_free_exports', 'export_administration_act_south_africa']
df = pd.read_csv('data.csv', names = headers_all)
df = df.replace('y', 1).replace('n', 0).replace('?', np.nan)
df = df.fillna(df.mean())
df['intercept'] = 1.0
features = ['handicapped_infants',
'water_project_cost_sharing','adoption_of_the_budget_resolution','physician_fee_freeze','el_salvador_aid', 'religious_groups_in_schools','anti_satellite_test_ban','aid_to_nicraguan_contras', 'mx_missile', 'immigration','synfuels_corporation_cutback', 'education_spending', 'superfund_right_to_sue','crime', 'duty_free_exports', 'export_administration_act_south_africa']
X = df[features]
y = df['party']


# training with a decision tree
dt_clf = DecisionTreeClassifier()
scores = cross_validation.cross_val_score(dt_clf, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# accuracy is .94 (+/- 0.05)

# plotting and saving as a pdf
dot_data = StringIO()
tree.export_graphviz(dt_train, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("politicians.pdf")

# training with a random forest just to view the difference
rf_clf = RandomForestClassifier()
scores = cross_validation.cross_val_score(rf_clf, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# accuracy is .95 (+/- 0.04)

