"""
Title: Data Science Team Project
Class: 2021-1 Data Science
Member: 201533645 BAE SEOUNG JAE
        201533631 KIM DO KYOON
        201633841 LEE KANG UK
"""



# =========================== Library ===========================
### Step 1. Import the libraries
import numpy as np
import pandas as pd
from pandas._libs.missing import NA

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler

from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import seaborn as sns

# Opensource SW Contribution Project
import FindBestScaleModel as fbs



# =========================== Function ===========================
# ============================ Plots ============================
"""
Name: coefplot
Input: coef, column, title
Output: plt.show
Desc: This function shows plot the coefficient plot with column name
"""
def coefplot(coef, column, title):
    plt.title(title)
    plt.ylabel("Coef")
    plt.scatter(column, coef, alpha=1, label="predict")
    
    plt.show()



"""
Name: barplot
Input: data, title
Output: None
Desc: This function shows barplot about data
"""
def barplot(data, title):
    label = data.index

    N = len(data.index)
    index = np.arange(N)

    alpha = 0.5
    bar_width = 0.35

    p1 = plt.bar(index, data['Score'],
                 bar_width,
                 color='b',
                 alpha=alpha,
                 label='Score')

    p2 = plt.bar(index + bar_width, data['Cross Val Score'],
                 bar_width,
                 color='r',
                 alpha=alpha,
                 label='Cross_Vaildate')

    plt.title(title, fontsize=20)

    plt.ylabel('Accuracy', fontsize=18)
    plt.xlabel('Model', fontsize=18)
    plt.xticks(index, label, fontsize=10)

    plt.legend((p1[0], p2[0]), ('Score', 'Cross_Vaildate'), fontsize=10, loc='upper right')

    plt.show()



"""
Name: XYPlot
Input: X(numpy), y(numpy), fitted model, title
Output: None
Desc: This function plot the x,y plot with y and predicted y value
"""
def xyplot(X, y, model, title):
    plt.figure(figsize=(18,8))
    xbar = np.arange(X.min(), X.max(), 0.01)
    ybar = model.predict(xbar.reshape(-1,1))
    plt.plot(xbar, ybar, "r", X, y, "bo")
    plt.axis([-5, 35, 0, 7])
    plt.title(title)

    plt.show()



"""
Name: heatmap
Input: df(dataframe), title
Output: None
Desc: This function shows heatmap about input dataframe
"""
def heatmap(df, title):
    corrmat = df.corr()
    feature = corrmat.index

    plt.figure(figsize=(10,10))
    ax = plt.axes()
    plt.title(title)
    g = sns.heatmap(df[feature].corr(), annot=True, cmap="RdYlGn", ax=ax)

    plt.show()





# =========================== Main ===========================
# Input all preprocessed dataframes
df = [
    pd.read_csv("location_10000.csv"),
    pd.read_csv("location_11000.csv"),
    pd.read_csv("location_12000.csv"),
    pd.read_csv("location_13000.csv"),
    pd.read_csv("location_14000.csv"),
    pd.read_csv("location_15000.csv"),
    pd.read_csv("location_16000.csv"),
    pd.read_csv("location_20000.csv"),
    pd.read_csv("location_30000.csv"),
    pd.read_csv("location_40000.csv"),
    pd.read_csv("location_41000.csv"),
    pd.read_csv("location_50000.csv"),
    pd.read_csv("location_51000.csv"),
    pd.read_csv("location_60000.csv"),
    pd.read_csv("location_61000.csv"),
    pd.read_csv("location_70000.csv"),
]

df_loc_list = pd.read_csv("location.csv")


# Set location name
location_name = df_loc_list['label'].to_numpy()


for i in range(0, len(df)):

    scaler = [
        None, # No Scale
        StandardScaler(),
        MinMaxScaler(),
        RobustScaler(),
        MaxAbsScaler(),
    ]

    print("========================")
    print("Analysis " + location_name[i] + " Data")

    #  Heatmap
    heatmap(df[i].drop(["date", "time"], axis=1), location_name[i])

    # Drop all features except temp
    X = df[i].drop(["Unnamed: 0", "date", "time", "confirmed","released", "deceased", "code", "most_wind_direction", "province", "target"], axis=1)
    # X = df[i].drop(["Unnamed: 0", "date", "time", "confirmed","released", "deceased", "code", "precipitation", "max_wind_speed", "most_wind_direction", "avg_relative_humidity", "province", "target"], axis=1)

    # Find best scale model 
    result = fbs.find(X, df[i]['target'], scaler, 10)
    print(result)

    # Extract best model
    model = result['best_model_']

    # If model is Linear regressor, print coefficient plot of each features
    if result['best_model_name_'] == 'LinearRegression':
        coefplot(model.coef_, X.columns, location_name[i] + " LinearRegression coef")
    print()


    # Drop all features except avg_temp
    X = df[i].drop(["Unnamed: 0", "date", "time", "confirmed","released", "deceased", "code", "precipitation", "max_wind_speed", "most_wind_direction", "avg_relative_humidity", "province", "min_temp", "max_temp", "target"], axis=1)

    # Find best scale model 
    result = fbs.find(X, df[i]['target'], scaler, 10)
    print(result)

    # Extract best model
    model = result['best_model_']

    # If model is Linear regressor, print coefficient plot of each features
    if result['best_model_name_'] == 'LinearRegression':
        coefplot(model.coef_, X.columns, location_name[i] + " LinearRegression coef")
    print()

    # Draw predicted data and real data plot
    xyplot(X['avg_temp'].to_numpy(), df[i]['target'].to_numpy(), model, location_name[i])
    
    
    
