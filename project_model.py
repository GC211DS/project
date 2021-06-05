"""
Title: Data Science Team Project
Class: 2021-1 Data Science
Member: 201533645 배성재
        201533631 김도균
        201633841 LEE KANG UK
"""

# =========================== Library ===========================
### Step 1. Import the libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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
Input: coef, column,title
Output: plt.show
Desc: This function plot the coef with column name
"""
def coefplot(results,column,title):
    plt.title(title)
    plt.ylabel("Coef")
    plt.scatter(column, results, alpha=1, label="predict")
    return plt.show()

"""
Name: barplot
Input: data, title
Output: None
Desc: This function barplot  about data
"""
def barplot(data,title):
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

    plt.legend((p1[0], p2[0]), ('Score', 'Cross_Vaildate'), fontsize=15)

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
    

# Set location name
location_name = ['Seoul', 'Busan', 'Daegu', 'Gwangju', 'Incheon', 'Daejeon', 'Ulsan',
 'Gyeonggi-do', 'Gangwon-do', 'Chungcheongbuk-do', 'Chungcheongnam-do',
 'Jeollabuk-do', 'Jeollanam-do', 'Gyeongsangbuk-do', 'Gyeongsangnam-do',
 'Jeju-do', 'Chunghceongbuk-do']

# Heatmap code
for i in range(0, len(df)):

    # scaling & drop data
    standardScaler = StandardScaler()

    # Scaled data
    # x_scaled = standardScaler.fit_transform(df[i].drop(['date','time','target'],axis = 1))
    # columns = df[0].drop(['date','time','target'],axis = 1).columns
    # df_scaled = pd.DataFrame(x_scaled, columns=columns)

    #  No scaled data
    df_scaled = df[i].drop(['date','time','target'], axis=1)

    df_scaled['target'] = df[i]['target']

    corrMat = df_scaled.corr()
    feature = corrMat.index

    # ax = plt.axes()

    # plt.title(location_name[i])
    # plt.figure(figsize=(11,11))
    # g = sns.heatmap(df_scaled[feature].corr(), annot=True, cmap="RdYlGn", ax = ax)
        
    # plt.show()



    # # 2. 온도를 비교하여 분석

    crossFold = 7

    
    # split data set
    print(df_scaled)
    
    # Drop all features except temp
    # X = df_scaled.drop(["Unnamed: 0", "confirmed","released", "deceased", "code", "precipitation", "max_wind_speed", "most_wind_direction", "avg_relative_humidity", "province", "target"], axis=1)

    # Drop all features except avg_temp
    X = df_scaled.drop(["Unnamed: 0", "confirmed","released", "deceased", "code", "precipitation", "max_wind_speed", "most_wind_direction", "avg_relative_humidity", "province", "min_temp", "max_temp", "target"], axis=1)

    print(X)
    x_train, x_test, y_train, y_test = train_test_split(X, df_scaled['target'], test_size=0.2, shuffle=True, random_state=34)

    # create dataframe for bar plot
    data = pd.DataFrame({'Score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        'Cross Val Score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})
    data = pd.DataFrame(data, index=['LinearRegression','PolynomialRegression','LogisticRegression','KNeighborsClassifier', 'DecisionTreeRegressor',
                  'BaggingRegressor','AdaBoostRegressor','GradientBoostingRegressor'])


    # TODO:
    # 각 지역별 모델 스코어 평균
    # 가장 평균이 높은 모델로 선택

    ##############################
    # linear regression
    model = LinearRegression()
    model.fit(x_train, y_train)
    print("LinearRegression score : " ,model.score(x_test, y_test))
    score = cross_val_score(model, x_test, y_test, cv=crossFold)
    print("cross_val score of LinearRegression", np.mean(score))
    title = location_name[i] + " LinearRegression coef"
    # coefplot(model.coef_, X.columns, title)
    data.loc['LinearRegression']['Score'] = model.score(x_test, y_test)
    data.loc['LinearRegression']['Cross Val Score'] = np.mean(score)



    ##############################
    # https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=tt2t2am1118&logNo=221182074409
    # Polynomial Regression
    poly_reg = PolynomialFeatures(degree = 7)
    X_poly = poly_reg.fit_transform(x_train)
    model = LinearRegression()
    model.fit(X_poly, y_train)
    print("PolynomialRegression score : ", model.score(poly_reg.fit_transform(x_test), y_test))
    score = cross_val_score(model, poly_reg.fit_transform(x_test), y_test, cv=crossFold)
    print("cross_val score of PolynomialRegression", np.mean(score))
    data.loc['PolynomialRegression']['Score'] = model.score(poly_reg.fit_transform(x_test), y_test)
    data.loc['PolynomialRegression']['Cross Val Score'] = np.mean(score)

    # Plot result
    plt.figure(figsize=(18,8))
    xbar = np.arange(-5, 35, 0.01)
    plt.plot(xbar, model.predict(poly_reg.fit_transform(xbar.reshape(-1,1))), "r", df_scaled['avg_temp'].to_numpy(), df_scaled['target'].to_numpy(), "bo")
    plt.axis([-5, 35, 0, 7])
    plt.title("PolynomialRegression: " + location_name[i])
    plt.show()



    ##############################
    # Logistic regression
    model = LogisticRegression()
    model.fit(x_train, y_train)
    print("LogisticRegression score : ", model.score(x_test, y_test))
    score = cross_val_score(model, x_test, y_test, cv=crossFold)
    print("cross_val score of LogisticRegression", np.mean(score))
    data.loc['LogisticRegression']['Score'] = model.score(x_test, y_test)
    data.loc['LogisticRegression']['Cross Val Score'] = np.mean(score)

    # TODO:
    # coef plot 상 avg_temp가 가장 의미있음
    # 이거만 가지고 학습 다시 진행
    # 나온 데이터 predict ->  plot

    # Plot result
    plt.figure(figsize=(18,8))
    xbar = np.arange(-5, 35, 0.01)
    ybar = model.predict(xbar.reshape(-1,1))
    plt.plot(xbar, ybar, "r", df_scaled['avg_temp'].to_numpy(), df_scaled['target'].to_numpy(), "bo")
    plt.axis([-5, 35, 0, 7])
    plt.title("LogisticRegression: " + location_name[i])
    plt.show()


    ##############################
    # KNeighborsClassifier
    model = KNeighborsClassifier()
    

    params = {
        'n_neighbors': np.arange(1,15)
    }
    grid_cv = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=crossFold, verbose=1)
    grid_cv.fit(x_train, y_train)
    print('GridSearchCV 최고 평균 정확도 수치: {:.4f}'.format(grid_cv.best_score_))
    print('GridSearchCV 최적 하이퍼파라미터: ', grid_cv.best_params_)
    # Conclusion: GridSearchCV 최적 하이퍼파라미터:  {'n_neighbors': 4}
    # avg_temp 만 있을 때 Conclusion: GridSearchCV 최적 하이퍼파라미터:  {'n_neighbors': 6}

    model = KNeighborsClassifier(n_neighbors = grid_cv.best_params_['n_neighbors'])
    model.fit(x_train, y_train)

    print("KNeighborsClassifier score : ", model.score(x_test, y_test))
    score = cross_val_score(model, x_test, y_test, cv=crossFold)
    print("cross_val score of KNeighborsClassifier", np.mean(score))
    data.loc['KNeighborsClassifier']['Score'] = model.score(x_test, y_test)
    data.loc['KNeighborsClassifier']['Cross Val Score'] = np.mean(score)

    # Plot result
    plt.figure(figsize=(18,8))
    xbar = np.arange(-5, 35, 0.01)
    ybar = model.predict(xbar.reshape(-1,1))
    plt.plot(xbar, ybar, "r", df_scaled['avg_temp'].to_numpy(), df_scaled['target'].to_numpy(), "go")
    plt.axis([-5, 35, 0, 7])
    plt.title("KNeighborsClassifier: " + location_name[i])
    plt.show()



    ##############################
    # Decision Tree Regressor
    model = DecisionTreeRegressor(max_depth=2)
    model.fit(x_train, y_train)
    # params = {
    #     'max_depth': [2, 4, 6, 8, 10]
    # }
    #
    # grid_cv = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=5, verbose=1)
    # grid_cv.fit(x_train, y_train)
    # print('GridSearchCV 최고 평균 정확도 수치: {:.4f}'.format(grid_cv.best_score_))
    # print('GridSearchCV 최적 하이퍼파라미터: ', grid_cv.best_params_)  GridSearchCV 최적 하이퍼파라미터:  {'max_depth': 2}
    print("DecisionTreeRegressor score : " ,model.score(x_test, y_test))
    score = cross_val_score(model, x_test, y_test, cv=crossFold)
    print("cross_val score of DecisionTreeRegressor", np.mean(score))
    data.loc['DecisionTreeRegressor']['Score'] = model.score(x_test, y_test)
    data.loc['DecisionTreeRegressor']['Cross Val Score'] = np.mean(score)



    ##############################
    # BaggingRegressor Ensemble
    model = BaggingRegressor( max_samples=0.5, max_features=0.5, bootstrap_features= True,
        base_estimator=DecisionTreeRegressor(max_depth=2), bootstrap=True)
    # params = {
    #     "max_samples": [0.5, 1.0],
    #     "max_features": [0.5, 1.0],
    #     "bootstrap": [True, False],
    #     "bootstrap_features": [True, False]}
    #
    #
    # grid_cv = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=5, verbose=1)
    # grid_cv.fit(x_train, y_train)
    # print('GridSearchCV 최고 평균 정확도 수치: {:.4f}'.format(grid_cv.best_score_))
    # print('GridSearchCV 최적 하이퍼파라미터: ', grid_cv.best_params_)
    # GridSearchCV 최적 하이퍼파라미터: {'bootstrap': True, 'bootstrap_features': True, 'max_features': 0.5, 'max_samples': 0.5}

    model.fit(x_train, y_train)
    print("BaggingRegressor score : " ,model.score(x_test, y_test))
    score = cross_val_score(model, x_test, y_test, cv=crossFold)
    print("cross_val score of BaggingRegressor", np.mean(score))
    data.loc['BaggingRegressor']['Score'] = model.score(x_test, y_test)
    data.loc['BaggingRegressor']['Cross Val Score'] = np.mean(score)



    ##############################
    # AdaBoostRegressor Ensemble
    model = AdaBoostRegressor(n_estimators=1,
        base_estimator=DecisionTreeRegressor(max_depth=2), learning_rate=1)
    # params = {
    #     'n_estimators': [1,2,3,4,5,6,7,8,9,10,50,100,200,300,400,500,1000]
    # }
    #
    # grid_cv = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=5, verbose=1)
    # grid_cv.fit(x_train, y_train)
    # print('GridSearchCV 최고 평균 정확도 수치: {:.4f}'.format(grid_cv.best_score_))
    # print('GridSearchCV 최적 하이퍼파라미터: ', grid_cv.best_params_) GridSearchCV 최적 하이퍼파라미터:  {'n_estimators': 1}
    model.fit(x_train, y_train)
    print("AdaBoostRegressor score : " ,model.score(x_test, y_test))
    score = cross_val_score(model, x_test, y_test, cv=crossFold)
    print("cross_val score of AdaBoostRegressor", np.mean(score))
    data.loc['AdaBoostRegressor']['Score'] = model.score(x_test, y_test)
    data.loc['AdaBoostRegressor']['Cross Val Score'] = np.mean(score)



    ##############################
    # GradientBoostingRegressor Ensemble
    model = GradientBoostingRegressor(n_estimators=1, learning_rate=0.01)

    # params = {
    #     'n_estimators': [1,2,3,4,5,6,7,8,9,10,50,100,200,300,400,500,1000],
    #     'learning_rate': [0.01,0.1,1]
    #
    # }
    #
    # grid_cv = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=5, verbose=1)
    # grid_cv.fit(x_train, y_train)
    # print('GridSearchCV 최고 평균 정확도 수치: {:.4f}'.format(grid_cv.best_score_))
    # print('GridSearchCV 최적 하이퍼파라미터: ', grid_cv.best_params_) GridSearchCV 최적 하이퍼파라미터:  {'learning_rate': 0.01, 'n_estimators': 1}
    model.fit(x_train, y_train)
    print("GradientBoostingRegressor score : " ,model.score(x_test, y_test))
    score = cross_val_score(model, x_test, y_test, cv=crossFold)
    print("cross_val score of GradientBoostingRegressor", np.mean(score))
    data.loc['GradientBoostingRegressor']['Score'] = model.score(x_test, y_test)
    data.loc['GradientBoostingRegressor']['Cross Val Score'] = np.mean(score)


    # barplot(data,location_name[i])




