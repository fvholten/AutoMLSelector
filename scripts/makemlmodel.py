from flaml import AutoML
import pandas as pd
import pickle
from datetime import datetime

df = pd.read_csv('data.csv')
X_train, y_train = df.iloc[:, :-1], df.iloc[:, -1]

automl = AutoML()

automl_settings = {
    "task": "multiclass",
    "time_budget": 2*60*60,
    "metric": 'accuracy',
    "log_file_name": 'automl{}.log'.format(datetime.now())    
}

automl.fit(X_train, y_train, **automl_settings)

with open('automl.pkl', 'wb') as f:
    pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
