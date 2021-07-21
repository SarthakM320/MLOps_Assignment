import dvc.api as dvc
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
import pickle
# from joblib import dump, load


with dvc.open(repo="https://github.com/SarthakM320/MLOps_Assignment", path="data/creditcard.csv", mode="r") as f:
    df = pd.read_csv(f)

train = df.sample(replace=False, frac=0.8)
test = df.sample(replace=False, frac=0.2)
train.to_csv(path_or_buf='../data/preprocessed/train.csv')
test.to_csv(path_or_buf='../data/preprocessed/test.csv')

labels = train.pop('Class')
test_label = test.pop('Class')

decision_tree = DecisionTreeClassifier(
    random_state=0, max_depth=4, criterion="entropy")

decision_tree = decision_tree.fit(train, labels)

with open("../models/model.pkl", "wb") as f:
    pickle.dump(decision_tree, f)

y_preds = decision_tree.predict(test)

f1score = sklearn.metrics.f1_score(y_true=test_label, y_pred=y_preds)
accuracy = sklearn.metrics.accuracy_score(y_true=test_label, y_pred=y_preds)

metrics_df = pd.Series({'F1_Score': f1score, 'Accuracy_score': accuracy})
metrics_df.to_json('../metrics/acc_f1.json')
