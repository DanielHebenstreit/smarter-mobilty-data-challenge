# Smarter Mobility Data Challenge
Group: `charging-boys`

This repository contains our solution to CodaLab's [Smarter Mobility Data Challenge](https://codalab.lisn.upsaclay.fr/competitions/7192).

## Setup

To reproduce the final submission, copy `train.csv` and `test.csv` into the `data` directory in the root of the repository and run

```{bash}
pip install -r requirements.txt
python ./main.py
```

The provided code was tested with Python 3.9.13.

## Approach

As detailed in the submitted PDF document, we train three models and compute a weighted average of their predictions as our final submission.

### XGBoost Regression Model

For each unique pair of `(station, target)`, we train an autoregressive model using [`XGBoost`](https://xgboost.readthedocs.io/en/stable/) and [`skforecast`](https://joaquinamatrodrigo.github.io/skforecast). We use the last 20 observations of a given target as input, as well as exogenous variables like sin/cos encoded time information (see code for details) and train each model with `n_estimators=100`.

### XGBoost Classification Model

We train a single non-autoregressive `XGBoost` model that takes a one-hot encoded station vector (91 dimensional) and the same time-encoding exogenous variables as our XGBoost regression model as input, and predicts a class integer between 0 and 19 (both inclusive), which corresponds to a configuration of a station. For example, class 0 might correspond to `Available=3, Charging=0, Passive=0, Other=0`.
