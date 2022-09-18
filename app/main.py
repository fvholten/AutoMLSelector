import pickle
import csv
import sys
import openml as oml
import pandas as pd
import logging
from pathlib import Path

def predict(automl, test_df):
    classes_to_proba = dict(zip(automl.classes_, automl.predict_proba(test_df)[0]))
    return classes_to_proba.get(automl.predict(test_df)[0]),automl.predict(test_df)[0]
    
curr_dir = Path(__file__).parent

with open(curr_dir.joinpath(r'_meta.csv')) as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  list_qualities = list()
  for line in csv_reader:
    list_qualities.append(line[0])
  list_qualities = list_qualities[1:]

with open(curr_dir.joinpath(r'automl.pkl'), "rb") as input_file:
  automl = pickle.load(input_file)

dataset_id = sys.argv[1]
dataset = oml.datasets.get_dataset(dataset_id)

logging.info('Loaded dataset:' + dataset.name)

t = dict()

for quality in list_qualities:
  t[quality] = [dataset.qualities.get(quality)]
test_df = pd.DataFrame(t)

proba_of_best, automl_tool = predict(automl, test_df)

print("For dataset:")
print(dataset)
print()
print()
print("The best-choice is: {automl_tool}".format(automl_tool=automl_tool))
print("-> with a probability of: {proba}".format(proba=proba_of_best))
