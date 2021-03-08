import pandas as pd
import sklearn
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from utils import tpr, fpr
import pprint
import numpy as np
np.set_printoptions(precision=2)

def load(drop_desc=True):
    # load pre-processed recidivism (COMPAS) data
    df = pd.read_csv('data_processed.csv')

    features = [f for f in list(df) if 'desc' not in f]

    df = df[features]

    return df


def train(train_data):
    features = [f for f in list(train_data) if f != 'two_year_recid']

    print(features)

    y_train = train_data['two_year_recid']
    X_train = train_data[features]

    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(X_train.values, y_train.values)

    return pipe


def evaluate(pipe, group, metric, data):

    evaluation_data = data.copy()
    assert metric in ['auc', 'brier', 'fpr']
    assert group in [0, 1, None]

    loss_functions = {'brier': brier_score_loss,
                      'auc': roc_auc_score,
                      'fpr': fpr}

    # group: 0 = white, 1 = black

    if group:
        # filter by group
        evaluation_data = evaluation_data[evaluation_data['race'] == group]

    # form X, y
    y = evaluation_data['two_year_recid']
    del evaluation_data['two_year_recid']
    X = evaluation_data

    # compute predictions
    y_hat =  pipe.predict_proba(X.values)[:, 1]

    # compute score
    loss = loss_functions[metric]
    score = loss(y, y_hat)

    return score


df = load(drop_desc=True)
pp = pprint.PrettyPrinter(indent=4)

# split into train test
train_df, test_df = train_test_split(df, test_size=0.33)


pipe = train(train_df)

results = {'black': {}, 'white': {}, 'overall': {}}
group_string_to_int = {'black': 0, 'white': 1, 'overall': None}

for metric, group in product(['auc', 'brier', 'fpr'], ['black', 'white', 'overall']):
    g = group_string_to_int[group]
    results[group][metric] = evaluate(pipe=pipe, group=g, metric=metric, data=test_df)



pp.pprint(results)

# so for this model,

imbalance_loss = results['black']['fpr'] -  results['white']['fpr']
loss = results['overall']['brier']

print(loss, imbalance_loss)





