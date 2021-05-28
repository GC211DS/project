"""
Title: Data Science Team Project
Class: 2021-1 Data Science
Member: 201533645 배성재
        (학번) 김도균
        201633841 LEE KANG UK
"""

# =========================== Library ===========================
### Step 1. Import the libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


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
# =========================== Main ===========================
# Read the test.csv files
df = pd.read_csv("./test.csv")

#scaling & drop data
standardScaler = StandardScaler()
standardScaler.fit(df.drop(['confirmed','date','Unnamed: 0','time'],axis = 1))
x_scaled = standardScaler.transform(df.drop(['confirmed','date','Unnamed: 0','time'],axis = 1))
#x_scaled = df.drop(['confirmed','date','Unnamed: 0','time'],axis = 1)
print(df.drop(['confirmed','date','Unnamed: 0','time'],axis = 1).columns)

#split data set
x_train, x_test, y_train, y_test = train_test_split(x_scaled, df['confirmed'], test_size=0.2, shuffle=True, random_state=34)


#linear regression
model = LinearRegression()
model.fit(x_train, y_train)
print("linear regression score : " ,model.score(x_test,y_test))
print(model.coef_)

coefplot(model.coef_, df.drop(['confirmed','date','Unnamed: 0','time'],axis = 1).columns)
scatterplot(y_test,model.predict(x_test),x_test)



