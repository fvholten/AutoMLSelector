import pickle
import csv
import sys
import openml as oml
import pandas as pd
import logging
from pathlib import Path

curr_dir = Path(__file__).parent

with open(curr_dir.joinpath(r'_meta.csv')) as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  list_qualities = list()
  for line in csv_reader:
    list_qualities.append(line[0])
  list_qualities = list_qualities[1:]

with open(curr_dir.joinpath(r'ml-models.pickle'), "rb") as input_file:
  ml_models = pickle.load(input_file)


dataset_id = sys.argv[1]
dataset = oml.datasets.get_dataset(dataset_id)

logging.info('Loaded dataset:' + dataset.name)

t = dict()

for quality in list_qualities:
  t[quality] = [dataset.qualities.get(quality)]
test_df = pd.DataFrame(t)

bestProbability = dict()
currentBest = 0
for key, value in ml_models.items():
  aml = value.get('model')
  probability = aml.predict_proba(test_df)[0]
  logging.info("{key}: {probability}".format(key=key, probability=probability))

  probability1 = float(probability[1])

  if probability1 == currentBest:
    bestProbability[key] = probability1

  if probability1 > currentBest:
    currentBest = probability1
    bestProbability.clear()
    bestProbability[key] = probability1

print("For dataset:\n\"{dataset}\"\n\nThe best-choice is: \"{automl_tool}\"\n-> with a probability of: {proba}".format(dataset=dataset, automl_tool=', '.join(bestProbability), proba=currentBest))
