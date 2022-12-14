# Smarter Mobility Data Challenge
Group: `charging-boys`

This repository contains our solution to CodaLab's [Smarter Mobility Data Challenge](https://codalab.lisn.upsaclay.fr/competitions/7192).

## Setup

To reproduce the final submission, copy `train.csv` and `test.csv` into the `data` directory in the root of the repository and run

```{bash}
pip install -r requirements.txt
```

Afterwards, you can run all the cells in the provided Jupyter Notebook `main.ipynb` in-order. It will train all the models and produce the final submission in a folder called `result`. 

The provided code was tested with Python 3.9.13 and 3.7.12.

## Training Time

On a single Intel Xeon® Gold 6130, it takes approximately 50 minutes to train all three models (combined).

## Approach

As detailed in the submitted PDF document, we train three models and compute a weighted average of their predictions as our final submission.

### XGBoost Regression Model

For each unique pair of `(station, target)`, we train an autoregressive model using [`XGBoost`](https://xgboost.readthedocs.io/en/stable/) and [`skforecast`](https://joaquinamatrodrigo.github.io/skforecast). We use the last 20 observations of a given target as input, as well as exogenous variables like sin/cos encoded time information (see code for details) and train each model with `n_estimators=100`.

### XGBoost Classification Model

We train a single non-autoregressive `XGBoost` model that takes a one-hot encoded station vector (91 dimensional) and the same time-encoding exogenous variables as our XGBoost regression model as input, and predicts a class integer between 0 and 19 (both inclusive), which corresponds to a configuration of a station. For example, class 0 might correspond to `Available=3, Charging=0, Passive=0, Other=0`.

### ARIMA Model

For each unique pair of `(station, target)`, we train an autoregressive ARIMA model with `order=(2,1,1)` (details are explained in the submitted PDF document).
This model only receives the last 2 values of the previously differenced time series as input (no exogenous variables) and outputs the next value for the particular target.

### Ensemble

Finally, we ensemble the post-processed outputs of the three models described above by computing a weighted average. The weights are `0.4`, `0.35` and `0.25` for the ARIMA, XGBoost Regression, and XGBoost Classification model, respectively.

## Deep Learning Approaches

Although not part of our final set of models, we evaluate state-of-the-art deep learning approaches, namely a [Transformer architecture](https://huggingface.co/docs/transformers/model_doc/time_series_transformer), an LSTM network, and a shallow and deep variant of a feed-forward neural network (i.e., a sequence of dense layers). For each architecture, we train a single model that takes the last `k` target values as input (some variants also receive time information as input) and predicts the next `m` values. We experiment with different values of `m, k` and find that all of these models fail to extrapolate the given time series in a meaningful manner. We hypothesize that the reasons are mainly two-fold: (1) Deep learning models require a sufficiently large and diverse data set to perform well, which was problematic for the given training data, and (2) complex models appear to learn the high-frequency random noise in the time series, which is counterproductive when predicting values for multiple weeks into the future.

## Final Note

Due to a bug in the code, we have unfortunately uploaded a unscaled solution as our final submission (i.e., the sum of the targets in each row is not necessarily 3). This has been fixed in this notebook, which is why you will see a difference in the solution file uploaded and the one produced by this notebook.