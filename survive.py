from sksurv.datasets import load_veterans_lung_cancer
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator




data_x, data_y = load_veterans_lung_cancer()

fig1 = plt.figure()
time, survival_prob = kaplan_meier_estimator(data_y["Status"], data_y["Survival_in_days"])
plt.step(time, survival_prob, where="post")
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")


fig2 = plt.figure()
data_x["Treatment"].value_counts()
for treatment_type in ("standard", "test"):
    mask_treat = data_x["Treatment"] == treatment_type
    time_treatment, survival_prob_treatment = kaplan_meier_estimator(
        data_y["Status"][mask_treat],
        data_y["Survival_in_days"][mask_treat])
    
    plt.step(time_treatment, survival_prob_treatment, where="post",
             label="Treatment = %s" % treatment_type)

plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")


fig3 = plt.figure()
for value in data_x["Celltype"].unique():
    mask = data_x["Celltype"] == value
    time_cell, survival_prob_cell = kaplan_meier_estimator(data_y["Status"][mask],
                                                           data_y["Survival_in_days"][mask])
    plt.step(time_cell, survival_prob_cell, where="post",
             label="%s (n = %d)" % (value, mask.sum()))

plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")


## Implementing the Cox proportional hazard model 
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis


## Transform the data to numeric type
data_x_numeric = OneHotEncoder().fit_transform(data_x)

## You can now literally fit the x and y after calling on the model
estimator = CoxPHSurvivalAnalysis()
## This gives us a vector of coefficients, one for each vairable, 
## where each value corresponds to the log hazard ratio.
estimator.fit(data_x_numeric, data_y)


''' 
Using the fitted model, we can predict a patient-specific 
survival function, by passing an appropriate data matrix to the 
estimator's predict_survival_function method .

First, let's create a set of four synthetic patients. 
'''

## To test if the predictions work, lets create a small 
## subsample to put through the estimator

x_new = pd.DataFrame.from_items(
    [(1, [65, 0, 0, 1, 60, 1, 0, 1]),
     (2, [65, 0, 0, 1, 60, 1, 0, 0]),
     (3, [65, 0, 1, 0, 60, 1, 0, 0]),
     (4, [65, 0, 1, 0, 60, 1, 0, 1])],
     columns=data_x_numeric.columns, orient='index')

## Try to predict the survival 
pred_surv = estimator.predict_survival_function(x_new)
for i, c in enumerate(pred_surv):
    plt.step(c.x, c.y, where="post", label="Sample %d" % (i + 1))
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")

## This will give you a survival curve plot of each of the samples that you
## put in x_new 


## Time to measure the performance of the fit/ estimator
## you can use it from the estimator itself

estimator.score(data_x_numeric, data_y)

## Determining the most predictive parameters
import numpy as np

def fit_and_score_features(X, y):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis()
    for j in range(n_features):
        Xj = X[:, j:j+1]
        m.fit(Xj, y)
        scores[j] = m.score(Xj, y)
    return scores

scores = fit_and_score_features(data_x_numeric.values, data_y)
pd.Series(scores, index=data_x_numeric.columns).sort_values(ascending=False)

from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

pipe = Pipeline([('encode', OneHotEncoder()),
                 ('select', SelectKBest(fit_and_score_features, k=3)),
                 ('model', CoxPHSurvivalAnalysis())])

from sklearn.model_selection import GridSearchCV

param_grid = {'select__k': np.arange(1, data_x_numeric.shape[1] + 1)}
gcv = GridSearchCV(pipe, param_grid)
gcv.fit(data_x, data_y)

pd.DataFrame(gcv.cv_results_).sort_values(by='mean_test_score', ascending=False)


pipe.set_params(**gcv.best_params_)
pipe.fit(data_x, data_y)

encoder, transformer, final_estimator = [s[1] for s in pipe.steps]
pd.Series(final_estimator.coef_, index=encoder.encoded_columns_[transformer.get_support()])

