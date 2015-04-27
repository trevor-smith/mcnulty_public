
import pandas as pd
import numpy as np

headers_all = ['party', 'handicapped-infants', 'water-project-cost-sharing', 
'adoption-of-the-budget-resolution', 'physician-fee-freeze', 
'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban', 
'aid-to-nicraguan-contras', 'mx-missile', 'immigration', 
'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue', 
'crime', 'duty-free-exports', 'export-administration-act-south-africa']
df = pd.read_csv('data.csv', names = headers_all)

# converting the data
df = df.replace('y', 1).replace('n', 0).replace('?', np.nan)

# there are a lot of '?' in the data
# goal is to replace them with column mean
df = df.fillna(df.mean())

# split data into training and test
from sklearn.cross_validation import train_test_split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)


