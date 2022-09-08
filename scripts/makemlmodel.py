from flaml import AutoML
import pandas as pd
import pickle

df = pd.read_csv('scripts/training-data/data.csv')
X_train, y_train = df.iloc[:, :-1], df.iloc[:, -1]

with open(r"app/automl.pkl", "rb") as input_file:
  automl1 = pickle.load(input_file)

automl2 = AutoML()
automl2.fit(X_train, y_train, task="multiclass", time_budget=60*60, starting_points=automl1.best_config_per_estimator)

with open("automl.pkl", "wb") as f:
    pickle.dump(automl2, f, pickle.HIGHEST_PROTOCOL)
