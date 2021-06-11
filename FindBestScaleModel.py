
import numpy as np
from numpy.lib.polynomial import polymul
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

"""
Name: findBestScaleModel
Input: X, y, scaler, cv
Output: Result dictionary
Desc: Find the best combination of scaler and model
"""
def find(X, y, scaler, cv):

    result = {}

    resultModel = pd.DataFrame({'model': [None, None, None, None, None]})
    resultModelName = np.array([])
    resultScore = np.array([])

    for i in range(0, len(scaler)):

        if(scaler[i] != None):
            # scaling
            x_scaled = scaler[i].fit_transform(X)
        else:
            x_scaled = X


        # split data set
        x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, shuffle=True, random_state=34)

        # create dataframe for bar plot
        data = pd.DataFrame({
            'Score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'Cross Val Score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        })
        data = pd.DataFrame(data, index=['LinearRegression','LogisticRegression','KNeighborsClassifier', 'DecisionTreeRegressor',
                    'BaggingRegressor','AdaBoostRegressor','GradientBoostingRegressor'])
        # data = pd.DataFrame(data, index=['', 'LogisticRegression','KNeighborsClassifier', '', '' ])

        

        # linear regression
        output_model = linear(x_train, y_train, x_test, y_test, data, cv)
        linearModel = output_model
        

        # LogisticRegression
        output_model = logistic(x_train, y_train, x_test, y_test, data, cv)
        logisticModel = output_model


        # KNeighborsClassifier
        output_model =  kneighbor(x_train, y_train, x_test, y_test, data, cv)
        kneighborModel = output_model


        # DecisionTreeRegressor
        output_model = decision_tree(x_train, y_train, x_test, y_test, data, cv)
        decisiontreeModel = output_model


        # BaggingRegressor Ensemble
        output_model = bagging(x_train, y_train, x_test, y_test, data, cv)
        baggingModel = output_model

        # AdaBoostRegressor Ensemble
        output_model = adaboost(x_train,y_train,x_test,y_test,data, cv)
        adaboostModel = output_model


        # GradientBoostingRegressor Ensemble
        output_model = gradient(x_train, y_train, x_test, y_test, data, cv)
        gradientModel = output_model

        
        # Add best Model and score
        resultModelName = np.append(resultModelName, [data.idxmax(axis=0)[data.max(axis=0).idxmax()]])
        resultScore = np.append(resultScore, [data.loc[data.idxmax(axis=0)[data.max(axis=0).idxmax()], data.max(axis=0).idxmax()]])
        
        modelIdx = data.index.tolist().index(data.idxmax(axis=0)[data.max(axis=0).idxmax()])
        if modelIdx == 0:
            resultModel.iloc[i, 0] = linearModel
        elif modelIdx == 1:
            resultModel.iloc[i, 0] = logisticModel
        elif modelIdx == 2:
            resultModel.iloc[i, 0] = kneighborModel
        elif modelIdx == 3:
            resultModel.iloc[i, 0] = decisiontreeModel
        elif modelIdx == 4:
            resultModel.iloc[i, 0] = baggingModel
        elif modelIdx == 5:
            resultModel.iloc[i, 0] = adaboostModel
        elif modelIdx == 6:
            resultModel.iloc[i, 0] = gradientModel


    result = {
        'best_scaler_' : scaler[resultScore.argmax()],
        'best_model_name_' : resultModelName[resultScore.argmax()],
        'best_model_' : resultModel.iloc[resultScore.argmax(),0],
        'best_score_' : resultScore.max(),
    }

    return result




# =========================== Leaner ===========================
"""
Name: linear
Input: x_train,y_train,x_test,y_test,data,cv
Output: model
Desc: This function train LinearRegression model and predict
"""
def linear(x_train,y_train,x_test,y_test,data,cv):
    model = LinearRegression()
    model.fit(x_train, y_train)
    # print("LinearRegression score : ", model.score(x_test, y_test))
    score = cross_val_score(model, x_test, y_test, cv=cv)
    # print("cross_val score of LinearRegression", np.mean(score))

    data.loc['LinearRegression']['Score'] = model.score(x_test, y_test)
    data.loc['LinearRegression']['Cross Val Score'] = np.mean(score)

    return model



"""
Name: logistic
Input: x_train,y_train,x_test,y_test,data,cv
Output: model
Desc: This function train LogisticRegression model and predict
"""
def logistic(x_train,y_train,x_test,y_test,data,cv):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    # print("LogisticRegression score : ", model.score(x_test, y_test))
    score = cross_val_score(model, x_test, y_test, cv=cv)
    # print("cross_val score of LogisticRegression", np.mean(score))
    data.loc['LogisticRegression']['Score'] = model.score(x_test, y_test)
    data.loc['LogisticRegression']['Cross Val Score'] = np.mean(score)

    return model



"""
Name: kneighbor
Input: x_train,y_train,x_test,y_test,data,cv
Output: model
Desc: This function train KNeighborsClassifier model and predict with GridSearchCV
"""
def kneighbor(x_train,y_train,x_test,y_test,data,cv):
    model = KNeighborsClassifier()

    params = {
        'n_neighbors': np.arange(1, 15)
    }
    grid_cv = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=cv, verbose=1)
    grid_cv.fit(x_train, y_train)
    # print('GridSearchCV 최고 평균 정확도 수치: {:.4f}'.format(grid_cv.best_score_))
    # print('GridSearchCV 최적 하이퍼파라미터: ', grid_cv.best_params_)
    # Conclusion: GridSearchCV 최적 하이퍼파라미터:  {'n_neighbors': 4}
    # avg_temp 만 있을 때 Conclusion: GridSearchCV 최적 하이퍼파라미터:  {'n_neighbors': 6}

    model = KNeighborsClassifier(n_neighbors=grid_cv.best_params_['n_neighbors'])
    
    # TODO: 기록용
    global knnEnsModel
    knnEnsModel = model

    model.fit(x_train, y_train)

    # print("KNeighborsClassifier score : ", model.score(x_test, y_test))
    score = cross_val_score(model, x_test, y_test, cv=cv)
    # print("cross_val score of KNeighborsClassifier", np.mean(score))
    data.loc['KNeighborsClassifier']['Score'] = model.score(x_test, y_test)
    data.loc['KNeighborsClassifier']['Cross Val Score'] = np.mean(score)

    return model



"""
Name: decisionTree
Input: x_train,y_train,x_test,y_test,data,cv
Output: model
Desc: This function train DecisionTreeRegressor model and predictwith GridSearchCV
"""
def decision_tree(x_train,y_train,x_test,y_test,data,cv):
    global desEnsModel

    model = DecisionTreeRegressor()
    params = {
        'max_depth': np.arange(1, 10)
    }

    grid_cv = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=cv, verbose=1)
    grid_cv.fit(x_train, y_train)
    # print('GridSearchCV 최고 평균 정확도 수치: {:.4f}'.format(grid_cv.best_score_))
    # print('GridSearchCV 최적 하이퍼파라미터: ', grid_cv.best_params_)  GridSearchCV 최적 하이퍼파라미터:  {'max_depth': 2}
    model = DecisionTreeRegressor(max_depth=grid_cv.best_params_['max_depth'])
    desEnsModel = model

    model.fit(x_train, y_train)
    # print("DecisionTreeRegressor score : ", model.score(x_test, y_test))
    score = cross_val_score(model, x_test, y_test, cv=cv)
    # print("cross_val score of DecisionTreeRegressor", np.mean(score))
    data.loc['DecisionTreeRegressor']['Score'] = model.score(x_test, y_test)
    data.loc['DecisionTreeRegressor']['Cross Val Score'] = np.mean(score)

    return model



"""
Name: bagging
Input: x_train,y_train,x_test,y_test,data,cv
Output: model
Desc: This function train BaggingRegressor model and predict with GridSearchCV
"""
def bagging(x_train,y_train,x_test,y_test,data,cv):
    global knnEnsModel

    model = BaggingRegressor(max_samples=0.5, max_features=0.5, bootstrap_features=True,
                             base_estimator=knnEnsModel, bootstrap=True)
    params = {
        "max_samples": [0.5, 1.0],
        "max_features": [0.5, 1.0],
        "bootstrap": [True, False],
        "bootstrap_features": [True, False]}

    grid_cv = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=cv, verbose=1)
    grid_cv.fit(x_train, y_train)
    # print('GridSearchCV 최고 평균 정확도 수치: {:.4f}'.format(grid_cv.best_score_))
    # print('GridSearchCV 최적 하이퍼파라미터: ', grid_cv.best_params_)
    # GridSearchCV 최적 하이퍼파라미터: {'bootstrap': True, 'bootstrap_features': True, 'max_features': 0.5, 'max_samples': 0.5}
    model = BaggingRegressor(max_samples=grid_cv.best_params_['max_samples'],
                             max_features=grid_cv.best_params_['max_features'],
                             bootstrap_features=grid_cv.best_params_['bootstrap_features'],
                             base_estimator=knnEnsModel,
                             bootstrap=grid_cv.best_params_['bootstrap'])

    model.fit(x_train, y_train)
    # print("BaggingRegressor score : ", model.score(x_test, y_test))
    score = cross_val_score(model, x_test, y_test, cv=cv)
    # print("cross_val score of BaggingRegressor", np.mean(score))
    data.loc['BaggingRegressor']['Score'] = model.score(x_test, y_test)
    data.loc['BaggingRegressor']['Cross Val Score'] = np.mean(score)
    
    return model



"""
Name: adaboost
Input: x_train,y_train,x_test,y_test,data,cv
Output: model
Desc: This function train AdaBoostRegressor model and predict with GridSearchCV
"""
def adaboost(x_train,y_train,x_test,y_test,data,cv):
    global desEnsModel

    model = AdaBoostRegressor(n_estimators=1,
                              base_estimator=desEnsModel)
    params = {
        'n_estimators':[1,5,10,20,30,40,50,100,200,300,400,500,1000]
    }

    grid_cv = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=cv, verbose=1)
    grid_cv.fit(x_train, y_train)
    # print('GridSearchCV 최고 평균 정확도 수치: {:.4f}'.format(grid_cv.best_score_))
    # print('GridSearchCV 최적 하이퍼파라미터: ', grid_cv.best_params_) GridSearchCV 최적 하이퍼파라미터:  {'n_estimators': 1}
    model = AdaBoostRegressor(n_estimators=grid_cv.best_params_['n_estimators'])

    model.fit(x_train, y_train)
    # print("AdaBoostRegressor score : ", model.score(x_test, y_test))
    score = cross_val_score(model, x_test, y_test, cv=cv)
    # print("cross_val score of AdaBoostRegressor", np.mean(score))
    data.loc['AdaBoostRegressor']['Score'] = model.score(x_test, y_test)
    data.loc['AdaBoostRegressor']['Cross Val Score'] = np.mean(score)

    return model



"""
Name: gradient
Input: x_train,y_train,x_test,y_test,data,cv
Output: model
Desc: This function train GradientBoostingRegressor model and predict with GridSearchCV
"""
def gradient(x_train,y_train,x_test,y_test,data,cv):
    model = GradientBoostingRegressor(n_estimators=1, learning_rate=0.01)

    params = {
        'n_estimators': [1,2,3,4,5,6,7,8,9,10,50,100,200,300,400,500,1000],
        'learning_rate': [0.01,0.1,1]

    }

    grid_cv = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=cv, verbose=1)
    grid_cv.fit(x_train, y_train)
    # print('GridSearchCV 최고 평균 정확도 수치: {:.4f}'.format(grid_cv.best_score_))
    # print('GridSearchCV 최적 하이퍼파라미터: ', grid_cv.best_params_) GridSearchCV 최적 하이퍼파라미터:  {'learning_rate': 0.01, 'n_estimators': 1}
    model = GradientBoostingRegressor(n_estimators=grid_cv.best_params_['n_estimators'],
                                      learning_rate=grid_cv.best_params_['learning_rate'])

    model.fit(x_train, y_train)
    # print("GradientBoostingRegressor score : ", model.score(x_test, y_test))
    score = cross_val_score(model, x_test, y_test, cv=cv)
    # print("cross_val score of GradientBoostingRegressor", np.mean(score))
    data.loc['GradientBoostingRegressor']['Score'] = model.score(x_test, y_test)
    data.loc['GradientBoostingRegressor']['Cross Val Score'] = np.mean(score)

    return model



## KNN Ensemble variables
knnEnsModel = ""
desEnsModel = ""