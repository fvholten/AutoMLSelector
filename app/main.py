import pickle , csv , sys , logging , pandas , openml, pathlib

def predict(ml_models, test_df):
  classes_to_proba = {k: v.get('model').predict_proba(test_df)[0][1] for k, v in ml_models.items()}
  best = max(classes_to_proba, key=classes_to_proba.get)
  return classes_to_proba.get(best), best

curr_dir = pathlib.Path(__file__).parent

with open(curr_dir.joinpath(r'_meta.csv')) as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  list_qualities = [line[0] for line in csv_reader][1:]

with open(curr_dir.joinpath(r'ml-models.pickle'), "rb") as input_file:
  ml_models = pickle.load(input_file)

dataset = openml.datasets.get_dataset(dataset_id=sys.argv[1])
logging.info('Loaded dataset:' + dataset.name)

test_df = pandas.DataFrame({ quality : [dataset.qualities.get(quality)] for quality in list_qualities})

proba_of_best, automl_tool = predict(ml_models, test_df)

print("For dataset:")
print(dataset)
print()
print("The best-choice is: {}".format(automl_tool))
print("-> with a probability of: {}".format(proba_of_best))
