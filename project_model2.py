"""
Title: Data Science Team Project
Class: 2021-1 Data Science
Member: 201533645 배성재
        201533631 김도균
        201633841 LEE KANG UK
"""

# =========================== Library ===========================
### Step 1. Import the libraries
from numpy import mod
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns


# =========================== Function ===========================
"""
Name: drop_before_2020
Input: weather.csv dataframe
Output: None
Desc: This function drops what the date value is before 2020.
"""
def linear(x,y):
    linearReg = LinearRegression()
    linearReg.fit(x, y)
    return linearReg

"""
Name: coefplot
Input: coef, column
Output: plt.show
Desc: This function plot the coef with column name
"""
def coefplot(results,column):
    plt.title("Coef about column")
    plt.ylabel("Coef")
    plt.scatter(column, results, alpha=1, label="predict")
    return plt.show()

"""
Name: scatterplot
Input: y_test, predict by fitted model, dataset
Output: None
Desc: This function plot scatter plot about data
"""
def scatterplot(y,predict,data):

    plt.scatter(data[:, 6], predict, alpha=0.4, label="predict")
    plt.scatter(data[:, 6], y, alpha=0.4, label="Test data")
    plt.legend()
    plt.xlabel("precipitation")
    plt.ylabel("Confirmed")
    plt.title("Confirmed & Precipitation")
    plt.show()



    plt.scatter(data[:, 5], predict, alpha=0.4, label="predict")
    plt.scatter(data[:, 5], y, alpha=0.4, label="Test data")
    plt.legend()
    plt.xlabel("Max_temp")
    plt.ylabel("Confirmed")
    plt.title("Confirmed & Max_temp")
    plt.show()

    plt.scatter(data[:, 1], predict, alpha=0.4, label="predict")
    plt.scatter(data[:, 1], y, alpha=0.4, label="Test data")
    plt.legend()
    plt.xlabel("deceased")
    plt.ylabel("Confirmed")
    plt.title("Confirmed & deceased")
    plt.show()
    return




# 지역별로 나눴던 csv 파일들을 dataframe으로 변환
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
    









# Heatmap code
for i in range(0, len(df)):

    # scaling & drop data
    standardScaler = StandardScaler()
    # standardScaler.fit(df.drop(['confirmed','date','time'],axis = 1))
    x_scaled = standardScaler.fit_transform(df[i].drop(['confirmed','date','time'],axis = 1))
    #x_scaled = df.drop(['confirmed','date','Unnamed: 0','time'],axis = 1)
    columns = df[0].drop(['confirmed','date','time'],axis = 1).columns

    df_scaled = pd.DataFrame(x_scaled, columns=columns)
    df_scaled['confirmed'] = df[i]['confirmed']

    corrMat = df_scaled.corr()
    feature = corrMat.index

    ax = plt.axes()
    ax.set_title(i)
    
    plt.figure(figsize=(11,11))
    g = sns.heatmap(df_scaled[feature].corr(), annot=True, cmap="RdYlGn", ax = ax)
        
    # plt.show()

    
    # split data set
    # x_train, x_test, y_train, y_test = train_test_split(df_scaled_10000, df_10000['confirmed'], test_size=0.2, shuffle=True, random_state=34)
    # # linear regression
    # model = LinearRegression()
    # model.fit(x_train, y_train)
    # print("linear regression score : " ,model.score(x_test,y_test))
    # print(model.coef_)

    # # 2. 온도를 비교하여 분석

    
    # split data set
    print(df_scaled)
    X = df_scaled.drop(["confirmed", "Unnamed: 0", "released", "deceased", "code", "precipitation", "max_wind_speed", "most_wind_direction", "avg_relative_humidity", "province"], axis=1)

    print(X)
    x_train, x_test, y_train, y_test = train_test_split(X, df_scaled['confirmed'], test_size=0.2, shuffle=True, random_state=34)

    #linear regression
    model = LinearRegression()
    model.fit(x_train, y_train)
    print("LinearRegression score : " ,model.score(x_test, y_test))
    # print(model.coef_)

    # Decision Tree Regressor
    model = DecisionTreeRegressor(max_depth=2)
    model.fit(x_train, y_train)
    print("DecisionTreeRegressor score : " ,model.score(x_test, y_test))

    # BaggingRegressor Ensemble
    model = BaggingRegressor(n_estimators=10000, max_samples=1, max_features=1,
        base_estimator=DecisionTreeRegressor(max_depth=2), bootstrap=True)

    model.fit(x_train, y_train)
    print("BaggingRegressor score : " ,model.score(x_test, y_test))

    # AdaBoostRegressor Ensemble
    model = AdaBoostRegressor(n_estimators=10000, 
        base_estimator=DecisionTreeRegressor(max_depth=2), learning_rate=1)

    model.fit(x_train, y_train)
    print("AdaBoostRegressor score : " ,model.score(x_test, y_test))

    # GradientBoostingRegressor Ensemble
    model = GradientBoostingRegressor(n_estimators=10000, learning_rate=0.01)

    model.fit(x_train, y_train)
    print("GradientBoostingRegressor score : " ,model.score(x_test, y_test))

    plt.show()

    # coefplot(model.coef_, df_scaled.drop(['confirmed','date','Unnamed: 0','time'],axis = 1).columns)
    # scatterplot(y_test,model.predict(x_test),x_test)



