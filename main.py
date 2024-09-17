import pandas as pd

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.inspection import DecisionBoundaryDisplay

import matplotlib.pyplot as plt

df = pd.read_csv('data/Hive17.csv', sep=";")
df = df.dropna()
print(df.head())

X = df[["T17", "RH17"]].values
estimators = [
    OneClassSVM(nu=0.1, gamma=0.05).fit(X),
    IsolationForest(n_estimators=100).fit(X)
]

for estimator in estimators:
    disp = DecisionBoundaryDisplay.from_estimator(
        estimator,
        X,
        response_method="decision_function",
        plot_method="contour",
        xlabel="Temperature", ylabel="Humidity",
        levels=[0],
    )
    disp.ax_.scatter(X[:, 0], X[:, 1])
    disp.ax_.set_title(
        f"Decision boundary using {estimator.__class__.__name__}"
    )
    plt.show()