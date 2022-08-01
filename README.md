# AutoMLSelector

## How to use the tool

Run with:
```
python3 app/main.py ${OPENML_DATASET_ID}
```

## Results:

The results will be printed to the console. 
The app will print the dataset information followed by the recommended AutoML Tool to use.

Possible AutoML tools are:

- [ATM](https://hdi-project.github.io/ATM/)
- [auto-sklearn](https://automl.github.io/auto-sklearn/master/#)
- [AutoGluon](https://auto.gluon.ai)
- [FLAML](https://microsoft.github.io/FLAML/)
- [H2O AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
- [Hyperopt](http://hyperopt.github.io/hyperopt/)
- [rminer AutoML](https://cran.r-project.org/web/packages/rminer/rminer.pdf)
- [TPOT](http://automl.info/tpot/)
- [TransmogrifAI](https://transmogrif.ai)

## Folder  scripts

This folder contains scripts which have been used to create the ML-Model.
